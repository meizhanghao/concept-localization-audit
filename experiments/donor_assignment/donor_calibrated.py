"""Bound donor sensitivity while holding annotation-point agreement fixed.

Run from main_python using python -B -m concept_null_audit.donor_calibrated.
"""

from itertools import permutations
import json
import time
import sys

from .donor_assignment import (
    CASES, CUB_SIDE, EPS, checked_strata, fingerprint, hit_matrix,
    cub_window_geometry, jaccard_distance, mask_window_geometry, matched_assignment,
    mean_interval, options, read_arrays, sign_radius, write_json,
)

import numpy as np
from scipy.optimize import linprog


def solve_stratum(objective, annotation):
    """Maximize integer hit cost with a fixed mean annotation-transfer hit rate.

    The primal mixes at most two derangements in this stratum. Query weights
    are constant here. A supporting line in the two-dimensional image of the
    assignment polytope certifies optimality without a dense linear program.
    """
    n = len(objective)
    row = np.arange(n)
    denominator = n - 1
    target_numerator = int(annotation.sum() - np.trace(annotation))
    def vertex(cost):
        columns = matched_assignment(-cost)
        return (
            int(annotation[row, columns].sum()),
            int(objective[row, columns].sum()),
            columns,
        )

    central = vertex(objective)
    remainder = central[0] * denominator - target_numerator
    if remainder == 0:
        return {"mean": float(central[1]), "witness": (central, central), "mix_high": 0.0,
                "dual_gap": 0.0, "lambda": 0.0, "vertices_examined": 1}
    # A one-unit change in the annotation hit count dominates any possible
    # change in the unscaled objective at this coefficient.
    multiplier = 1 + n * int(np.ptp(objective))
    if remainder > 0:
        low, high = vertex(objective - multiplier * annotation), central
    else:
        low, high = central, vertex(objective + multiplier * annotation)
    if not low[0] * denominator <= target_numerator <= high[0] * denominator or low[0] >= high[0]:
        raise AssertionError("The uniform annotation rate must be feasible.")
    examined = 2
    for _ in range(n + 1):
        delta_hits = high[0] - low[0]
        delta_value = high[1] - low[1]
        candidate = vertex(delta_hits * objective - delta_value * annotation)
        examined += 1
        gain = delta_hits * (candidate[1] - low[1]) - delta_value * (candidate[0] - low[0])
        if gain < 0:
            raise AssertionError("Supporting assignment scored below a feasible endpoint.")
        if gain == 0:
            mixture_denominator = denominator * delta_hits
            mixture_numerator = target_numerator - low[0] * denominator
            fraction = mixture_numerator / mixture_denominator
            primal = (mixture_denominator - mixture_numerator) * low[1] + mixture_numerator * high[1]
            dual = mixture_denominator * candidate[1] + delta_value * (target_numerator - candidate[0] * denominator)
            if primal != dual:
                raise AssertionError("Integer primal and dual certificates differ.")
            value = primal / mixture_denominator
            lam = delta_value / delta_hits
            return {"mean": float(value), "witness": (low, high), "mix_high": fraction,
                    "dual_gap": 0.0, "lambda": lam, "vertices_examined": examined}
        if not low[0] < candidate[0] < high[0]:
            raise AssertionError("Supporting vertex must narrow the calibration bracket.")
        if candidate[0] * denominator <= target_numerator:
            low = candidate
        else:
            high = candidate
    raise RuntimeError("Calibration search did not terminate.")


def check_small_assignments():
    """Independently solve the mixture LP over every small derangement."""
    rng = np.random.default_rng(2026112602)
    checked, max_error = 0, 0.0
    for n in range(2, 7):
        row = np.arange(n)
        choices = np.array([p for p in permutations(row) if np.all(row != p)])
        for _ in range(10):
            score = rng.integers(-5, 6, size=(n, n))
            annotation = rng.integers(0, 2, size=(n, n))
            labels = annotation[row, choices].sum(axis=1)
            costs = score[row, choices].sum(axis=1)
            reference = (annotation.sum() - np.trace(annotation)) / (n - 1)
            target = [1, reference]
            constraints = np.stack([np.ones(len(costs)), labels])
            lp = linprog(-costs, A_eq=constraints, b_eq=target, bounds=(0, None), method="highs")
            optimum = solve_stratum(score, annotation)
            if not lp.success:
                raise AssertionError(lp.message)
            error = abs(optimum["mean"] + lp.fun)
            np.testing.assert_allclose(error, 0.0, atol=EPS, rtol=0)
            max_error = max(max_error, error)
            checked += 1
    return {"enumerated_linear_programs": checked, "maximum_objective_error": max_error}


def annotation_peaks(context, inputs, context_name):
    if context_name == "FB_mask_context":
        return read_arrays(inputs / "FB_mask_annotation.npz")["peaks"]
    points = context["points_xy"]
    valid = (points >= 0).all(axis=-1) & (points <= CUB_SIDE).all(axis=-1)
    if not valid.any(axis=1).all():
        raise ValueError("Expected at least one valid CUB keypoint in each query.")
    choice = valid.argmax(axis=1)
    return np.minimum(points[np.arange(len(points)), choice], CUB_SIDE - 1)[None]


def freeze_followup(output):
    path = output / "calibration_protocol.json"
    protocol = {
        "status": "Exploratory follow-up designed after geometry-only matching results.",
        "reason": "Direct geometry matching also reduced the annotation-point positive control. Hold its uniform donor hit rate exactly within each eligible stratum.",
        "annotation_reference": "Existing annotation-point peaks on FunnyBirds. First valid annotated keypoint clipped to the pixel grid in each CUB query.",
        "fixed": "Query, part/attribute, recipient window, trained and random peaks, eligible edges, macro weights and donor marginals. The annotation constraint applies to each stratum separately.",
        "target": "Minimum/maximum finite-cohort crossed excess and fraction of worst-case mixing needed to reach zero. Also compare CAM and gradients with one common assignment.",
        "baseline": "Exact uniform all-other donor, equal occurrence of every donor in its stratum.",
        "verification": "Recover nominal points from the hit matrices. Compare with exhaustive small-instance LP. Save witnesses to check feasibility and reported endpoints.",
        "paper_rule": "Calibration is useful if both originally well-supported datasets exhibit a reachable zero even when preserving the annotation-point control. Its uncertainty is over allowed assignments, not an estimate of population sign or causal faithfulness.",
        "baseline_protocol_sha256": fingerprint(output / "protocol.json"),
        "baseline_result_sha256": fingerprint(output / "results.json"),
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("The donor sensitivity follow-up protocol has changed.")
    else:
        write_json(path, protocol)


def freeze_geometry_followup(output):
    path = output / "calibrated_geometry_protocol.json"
    protocol = {
        "scope": "Exploratory follow-up to the geometry match and calibrated extremum analyses, before computing any geometry match that preserves the annotation control.",
        "objective": "Minimize mean window Jaccard distance, quantized to 1e-5, while preserving the exact uniform annotation-transfer hit rate in each stratum.",
        "reason": "Test whether a score-independent schedule can reduce the score without sacrificing the annotation-point control.",
        "randomization": "Use the per-stratum optimal mixture of at most two assignments as expected donor hits. Do not sample outcomes or tune a distance threshold.",
        "criterion": "Record a changed verdict only if the original exact-uniform interval is above zero, the new interval reaches zero, and the paired difference interval is below zero. Evaluate all nine primary cases.",
        "intervals": "Pointwise paired image/head intervals over cached draws, conditional on cohorts, assignments and calibration. No population generalization from these reused cohorts.",
        "calibration_sha256": fingerprint(output / "calibration_protocol.json"),
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("The annotation-preserving geometry protocol has changed.")
    else:
        write_json(path, protocol)


def integer_contrast(hits):
    heads = len(hits) - 1
    if heads < 1:
        raise ValueError("A trained and at least one random head are required.")
    return hits[0].astype(np.int16) * heads - hits[1:].sum(axis=0), heads


def verify_mixture(solution, cost, annotation):
    """Recompute the objective and calibration directly from the two permutations."""
    row = np.arange(len(cost))
    a, b = (vertex[2] for vertex in solution["witness"])
    for permutation in (a, b):
        np.testing.assert_array_equal(np.sort(permutation), row)
        if np.any(permutation == row):
            raise AssertionError("A witness assigned a query to itself.")
    mix = solution["mix_high"]
    score = (1 - mix) * int(cost[row, a].sum()) + mix * int(cost[row, b].sum())
    anchor = (1 - mix) * int(annotation[row, a].sum()) + mix * int(annotation[row, b].sum())
    original_anchor = (annotation.sum() - np.trace(annotation)) / (len(cost) - 1)
    np.testing.assert_allclose(anchor, original_anchor, atol=EPS, rtol=0)
    np.testing.assert_allclose(score, solution["mean"], atol=EPS, rtol=EPS)
    return a, b, mix


def calibrated_examples():
    """Equal original four scores can coexist with different calibrated ranges."""
    n = 6
    rows = np.arange(n)
    shift = (rows + 1) % n
    anchor = np.eye(n, dtype=int)
    anchor[0, shift[0]] = 1
    anchor[rows[1:5], (rows[1:5] + 2) % n] = 1
    stable = np.eye(n, dtype=int)
    stable[:2, :] = 1
    sensitive = np.eye(n, dtype=int)
    sensitive[rows, shift] = 1
    sensitive[rows[:4], (rows[:4] + 3) % n] = 1
    examples = {}
    for name, cost in (("stable", stable), ("sensitive", sensitive)):
        nominal = 1 - (cost.sum() - np.trace(cost)) / (n * (n - 1))
        maximum = solve_stratum(cost, anchor)
        minimum = solve_stratum(-cost, anchor)
        verify_mixture(maximum, cost, anchor)
        verify_mixture(minimum, -cost, anchor)
        bounds = [1 - maximum["mean"] / n, 1 + minimum["mean"] / n]
        expected_lower = 2/3 if name == "stable" else 0.0
        np.testing.assert_allclose([nominal, bounds[0]], [2/3, expected_lower], atol=EPS, rtol=0)
        examples[name] = {"S": 1.0, "G": nominal, "G_random": 0.0, "delta_G": nominal,
                          "calibrated_range": bounds, "annotation_gain": 5/6}
    return examples


def calibrated_cohort(context_name, names, metadata, inputs, keypoint_hit, baseline):
    context = read_arrays(inputs / f"{context_name}.npz")
    image, group = context["image"], context["group"]
    blocks = checked_strata(context, metadata)
    weights = 1 / (len(np.unique(group)) * np.bincount(group)[group])
    peaks = {name: read_arrays(inputs / f"{name}.npz")["peaks"] for name in names}
    anchor = annotation_peaks(context, inputs, context_name)
    window = None
    if metadata is not None:
        side = int(context["side"])
        window = np.unpackbits(context["windows_packed"], axis=1, count=side**2).astype(bool)

    ranking = "cam_minus_gradient"
    witness_columns = {key: np.empty((2, 2, len(image)), np.int32) for key in (*names, ranking)}
    witness_mixtures = {key: np.empty((2, len(blocks)), float) for key in witness_columns}
    values = {key: np.zeros(4) for key in witness_columns}  # own, uniform donor, min donor, max donor
    geometry_gains = {key: np.empty(peak.shape[:2]) for key, peak in peaks.items()}
    geometry_shift = {key: np.empty(peak.shape[:2]) for key, peak in peaks.items()}
    geometry_columns = np.empty((2, len(image)), np.int32)
    geometry_mixtures = np.empty(len(blocks))
    geometry_distance = 0.0
    errors, vertex_counts = [], []
    annotation_own, annotation_gain = 0.0, 0.0
    for index, block in enumerate(blocks):
        weight, count = weights[block[0]], len(block)
        if (weights[block] != weight).any():
            raise ValueError("The solver requires a constant query weight within each stratum.")
        label_cost = hit_matrix(anchor, block, context, window, keypoint_hit)[0].astype(np.int64)
        anchor_own = np.trace(label_cost)
        label_uniform = (label_cost.sum() - anchor_own) / (count - 1)
        annotation_own += weight * anchor_own
        annotation_gain += weight * (anchor_own - label_uniform)
        points = (cub_window_geometry(context["points_xy"][block]) if window is None
                  else mask_window_geometry(window[block]))
        distance = jaccard_distance(*points)
        geometry_cost = -np.rint(distance * 100000).astype(np.int64)
        choice = solve_stratum(geometry_cost, label_cost)
        rows = np.arange(count)
        low, high, mix = verify_mixture(choice, geometry_cost, label_cost)
        geometry_columns[:, block] = block[np.stack([low, high])]
        geometry_mixtures[index] = mix
        geometry_distance += weight * ((1 - mix) * distance[rows, low] + mix * distance[rows, high]).sum()
        contrasts = {}
        for key, peak in peaks.items():
            hits = hit_matrix(peak, block, context, window, keypoint_hit)
            contrasts[key] = integer_contrast(hits)
            own = hits[:, rows, rows]
            donor = (1 - mix) * hits[:, rows, low] + mix * hits[:, rows, high]
            uniform_hits = (hits.sum(axis=2) - own) / (count - 1)
            geometry_gains[key][:, block] = own - donor
            geometry_shift[key][:, block] = uniform_hits - donor
        cam, cam_heads = contrasts[names[0]]
        gradient, gradient_heads = contrasts[names[1]]
        contrasts[ranking] = (cam * gradient_heads - gradient * cam_heads, cam_heads * gradient_heads)
        for key, (cost, scale) in contrasts.items():
            minima = solve_stratum(-cost, label_cost)
            maxima = solve_stratum(cost, label_cost)
            own = np.trace(cost)
            donor_average = (cost.sum() - own) / (count - 1)
            values[key] += weight / scale * np.array([own, donor_average, -minima["mean"], maxima["mean"]])
            for direction, (solution, objective) in enumerate(((minima, -cost), (maxima, cost))):
                a, b, fraction = verify_mixture(solution, objective, label_cost)
                witness_columns[key][direction, 0, block] = block[a]
                witness_columns[key][direction, 1, block] = block[b]
                witness_mixtures[key][direction, index] = fraction
                errors.append(solution["dual_gap"])
                vertex_counts.append(solution["vertices_examined"])

    rows, witness = {}, {}
    for key, (own, nominal_donor, low_donor, high_donor) in values.items():
        nominal = own - nominal_donor
        low, high = own - high_donor, own - low_donor
        original = baseline["cases"][key] if key in baseline["cases"] else baseline["common_assignment_cam_minus_gradient"]
        nominal_reference = original["uniform"]
        if isinstance(nominal_reference, dict):
            nominal_reference = nominal_reference["mean"]
        np.testing.assert_allclose(nominal, nominal_reference, atol=EPS, rtol=0)
        if not original["exact_range"][0] - EPS <= low <= nominal + EPS <= high + 2*EPS:
            raise AssertionError("Calibrated lower bound must lie inside the original range.")
        if high > original["exact_range"][1] + EPS:
            raise AssertionError("Calibrated upper bound must lie inside the original range.")
        rows[key] = {"uniform": float(nominal), "exact_range": [float(low), float(high)],
                     "fraction_to_zero": sign_radius(nominal, low, high),
                     "lower_at_10_percent_mixture": float(.9*nominal + .1*low)}
        if key in names:
            head_draws = read_arrays(inputs / f"{key}.npz")["head_indices"]
            paired = (weights, group, image, context["weights"], head_draws)
            geometry_fixed = mean_interval(geometry_gains[key], *paired)
            shift = mean_interval(geometry_shift[key], *paired)
            rows[key]["calibrated_geometry"] = geometry_fixed
            rows[key]["calibrated_geometry_minus_uniform"] = shift
            rows[key]["calibrated_geometry_retires_positive_verdict"] = (
                original["uniform"]["image_head_ci95"][0] > 0
                and geometry_fixed["image_head_ci95"][0] <= 0
                and shift["image_head_ci95"][1] < 0
            )
        witness[f"{key}__permutations"] = witness_columns[key]
        witness[f"{key}__mixture"] = witness_mixtures[key]
    witness["stratum_by_query"] = np.empty(len(image), np.int32)
    for index, block in enumerate(blocks):
        witness["stratum_by_query"][block] = index
    witness["geometry_permutations"] = geometry_columns
    witness["geometry_mixture"] = geometry_mixtures
    return {
        "annotation_own_S": float(annotation_own), "fixed_annotation_gain": float(annotation_gain),
        "minimum_annotation_constraints": len(blocks),
        "maximum_certificate_gap": max(errors),
        "maximum_vertices_examined": max(vertex_counts),
        "calibrated_geometry_mean_window_distance": float(geometry_distance),
        "cases": rows,
    }, witness


def main():
    start = time.monotonic()
    repo, output = options(__doc__)
    sys.path.insert(0, str(repo))
    from audit.io import verify_inputs
    from audit.metrics import keypoint_hit

    verify_inputs(repo / "data")
    freeze_followup(output)
    freeze_geometry_followup(output)
    checks = check_small_assignments()
    checks["calibrated_examples"] = calibrated_examples()
    inputs = repo / "data/core_audit/data"
    fb_meta = read_arrays(repo / "experiments/renderer_validity/data/inputs.npz")
    baseline = json.loads((output / "results.json").read_text(encoding="utf-8"))["cohorts"]
    reports, witnesses = {}, {}
    for context_name, cases in CASES.items():
        metadata = fb_meta if context_name.startswith("FB") else None
        summary, witness = calibrated_cohort(context_name, cases, metadata, inputs, keypoint_hit, baseline[context_name])
        reports[context_name] = summary
        witnesses.update({f"{context_name}__{name}": array for name, array in witness.items()})
        print(f"Calibrated {context_name} in {time.monotonic()-start:.1f} s", flush=True)
    write_json(output / "calibrated_results.json", {"checks": checks, "cohorts": reports})
    np.savez_compressed(output / "calibrated_witnesses.npz", **witnesses)


if __name__ == "__main__":
    main()
