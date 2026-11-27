"""Compare stratum-mean and per-query donor calibration on common CUB support.

Run from main_python with python -B -m concept_null_audit.donor_query_calibration.
The audit uses released Loc-CBM peak and annotation caches. It fits no models.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
import sys

from .donor_assignment import (
    checked_strata, fingerprint, hit_matrix, options, read_arrays, sign_radius, write_json,
)
from .donor_calibrated import annotation_peaks, integer_contrast, solve_stratum, verify_mixture
from .donor_policy_details import exact_species_total

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

TOL = 1e-8


def query_calibrated_donor_totals(cost, annotation, allowed, target):
    """Return donor hit-contrast extrema after fixing each recipient's expected annotation hit."""
    n = len(cost)
    recipient, donor = np.where(allowed)
    edge = np.arange(len(recipient))
    # Each query supplies the calibrated mass to hit and miss edges separately.
    source = 2 * recipient + annotation[recipient, donor]
    matrix = coo_matrix(
        (np.ones(2 * len(edge)),
         (np.r_[source, 2 * n + donor], np.r_[edge, edge])),
        shape=(3 * n, len(edge)),
    ).tocsr()
    rhs = np.r_[np.column_stack((1 - target, target)).ravel(), np.ones(n)]
    values = cost[recipient, donor].astype(float)
    extrema = []
    residuals = []
    for direction in (1, -1):
        fit = linprog(direction * values, A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs")
        if not fit.success:
            raise RuntimeError(f"Per-query donor calibration failed: {fit.message}")
        residual = float(np.max(np.abs(matrix @ fit.x - rhs)))
        if residual > TOL or np.min(fit.x) < -TOL:
            raise AssertionError(f"Infeasible donor policy: residual {residual:g}.")
        extrema.append(float(values @ fit.x))
        residuals.append(residual)
    return extrema, max(residuals)


@lru_cache(None)
def integer_row_allocations(n, i):
    """Enumerate allocations of n-1 units to nonself donors in a small species block."""
    return tuple(shares[:i] + (0,) + shares[i:] for shares in product(range(n), repeat=n - 1)
                 if sum(shares) == n - 1)


def exact_species_block(cost, annotation):
    """Check the LP independently by integer transport in a 2–4 image block."""
    n = len(cost)
    if not 2 <= n <= 4:
        raise ValueError("The exact check assumes the frozen CUB split's four-image species cap.")
    # Multiplying masses by n-1 makes the split-source transport supplies integral.
    # A network transport polytope with integral margins has integral vertices.
    choices = []
    for i in range(n):
        hit = annotation[i].tolist()
        value = cost[i].tolist()
        hit_units = sum(hit) - hit[i]
        choices.append([(row, sum(k * v for k, v in zip(row, value)))
                        for row in integer_row_allocations(n, i)
                        if sum(k * b for k, b in zip(row, hit)) == hit_units])

    @lru_cache(None)
    def extremes(i, demand):
        if i == n:
            return (0, 0) if not any(demand) else None
        totals = []
        for allocation, value in choices[i]:
            remainder = tuple(available - used for available, used in zip(demand, allocation))
            if min(remainder) < 0:
                continue
            tail = extremes(i + 1, remainder)
            if tail is not None:
                totals.append((value + tail[0], value + tail[1]))
        if not totals:
            return None
        return min(lo for lo, _ in totals), max(hi for _, hi in totals)

    return tuple(Fraction(total, n - 1) for total in extremes(0, (n - 1,) * n))


def group_mean_donor_totals(cost, annotation, allowed, target):
    totals = []
    for direction in (-1, 1):
        objective = direction * cost
        result = solve_stratum(objective, annotation, allowed=allowed, anchor_target=target)
        verify_mixture(result, objective, annotation, allowed=allowed, anchor_target=target)
        totals.append(direction * result["mean"])
    return totals


def run(data, keypoint_hit):
    inputs = data / "core_audit/data"
    context_name = "new_CUB_context"
    paths = {
        "context": inputs / f"{context_name}.npz",
        "peaks": inputs / "new_CUB_native_LocCBM.npz",
        "species": data / "extensions/donor_species.npz",
    }
    context = read_arrays(paths["context"])
    image = context["image"]
    species = read_arrays(paths["species"])["species"][image]
    peaks = read_arrays(paths["peaks"])["peaks"]
    anchor = annotation_peaks(context, inputs, context_name)
    groups = checked_strata(context)
    classes = {f"{species_rule}_{calibration}": []
               for species_rule in ("all_species", "same_species")
               for calibration in ("group_mean", "per_query")}
    baselines = []
    support = []
    exact_blocks = 0
    max_residual = 0.0
    max_exact_error = 0.0
    for index, group in enumerate(groups, start=1):
        labels = species[group]
        group = group[(labels[:, None] == labels).sum(axis=1) >= 2]
        n = len(group)
        if not n:
            raise AssertionError("An attribute has no same-species recipient support.")
        support.extend(group.tolist())
        labels = species[group]
        nonself = ~np.eye(n, dtype=bool)
        same = nonself & (labels[:, None] == labels)
        annotation = hit_matrix(anchor, group, context, None, keypoint_hit)[0].astype(np.int64)
        cost, heads = integer_contrast(hit_matrix(peaks, group, context, None, keypoint_hit))
        target = (annotation * same).sum(axis=1) / same.sum(axis=1)
        target_total = exact_species_total(annotation, labels)
        np.testing.assert_allclose(target.sum(), float(target_total), atol=TOL, rtol=0)
        own = int(np.trace(cost))
        uniform = exact_species_total(cost, labels)
        norm = heads * n
        baseline = float((own - uniform) / norm)
        baselines.append(baseline)
        for rule, allowed in (("all_species", nonself), ("same_species", same)):
            coarse = group_mean_donor_totals(cost, annotation, allowed, target_total)
            strict, residual = query_calibrated_donor_totals(cost, annotation, allowed, target)
            max_residual = max(max_residual, residual)
            order = [coarse[0], strict[0], float(uniform), strict[1], coarse[1]]
            if np.any(np.diff(order) < -TOL):
                raise AssertionError("Ranges must be nested and contain the shared baseline.")
            classes[f"{rule}_group_mean"].append([(own - x) / norm for x in coarse[::-1]])
            classes[f"{rule}_per_query"].append([(own - x) / norm for x in strict[::-1]])
            if rule == "same_species":
                exact = [Fraction(0), Fraction(0)]
                for label in np.unique(labels):
                    block = np.flatnonzero(labels == label)
                    sub = np.ix_(block, block)
                    low, high = exact_species_block(cost[sub], annotation[sub])
                    exact[0] += low
                    exact[1] += high
                    exact_blocks += 1
                error = np.max(np.abs(np.array(exact, dtype=float) - strict))
                np.testing.assert_allclose(error, 0.0, atol=TOL, rtol=0)
                max_exact_error = max(max_exact_error, float(error))
        if index % 10 == 0 or index == len(groups):
            print(f"Checked {index}/{len(groups)} attributes.", flush=True)
    d0 = float(np.mean(baselines))
    comparisons = {}
    for name, group_bounds in classes.items():
        lower, upper = np.mean(group_bounds, axis=0).tolist()
        radius = sign_radius(d0, lower, upper)
        comparisons[name] = {"range": [lower, upper], "zero_reachability": radius}
    for level in ("group_mean", "per_query"):
        broad_low, broad_high = comparisons[f"all_species_{level}"]["range"]
        near_low, near_high = comparisons[f"same_species_{level}"]["range"]
        if not broad_low - TOL <= near_low <= d0 <= near_high <= broad_high + TOL:
            raise AssertionError("Same-species restriction must preserve the shared baseline and nest the range.")
    return {
        "scope": "Exploratory finite-cohort sensitivity on identical recipients, maps, reference heads and weights.",
        "baseline": "Uniform same-species nonself donors within each attribute. Attributes have equal weight.",
        "queries": len(support),
        "full_queries": len(image),
        "images": int(np.unique(image[support]).size),
        "attributes": len(groups),
        "uniform_excess": d0,
        "classes": comparisons,
        "checks": {
            "exact_species_blocks": exact_blocks,
            "max_exact_total_error": max_exact_error,
            "max_lp_feasibility_residual": max_residual,
        },
        "inputs_sha256": {str(path.relative_to(data).as_posix()): fingerprint(path)
                          for path in paths.values()},
    }


def main():
    repo, output = options(__doc__)
    sys.path.insert(0, str(repo))
    from audit.io import verify_inputs
    from audit.metrics import keypoint_hit

    count = verify_inputs(repo / "data")
    report = run(repo / "data", keypoint_hit)
    report["checks"]["inputs_verified"] = count
    write_json(output / "query_calibration.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
