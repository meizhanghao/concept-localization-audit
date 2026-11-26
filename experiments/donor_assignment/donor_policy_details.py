"""Recover the archived schedule scores and isolate CUB donor species eligibility."""

from fractions import Fraction
from itertools import permutations
import json
import sys

import numpy as np
from scipy.optimize import linprog

from .donor_assignment import CASES, EPS, checked_strata, fingerprint, hit_matrix, options, read_arrays, write_json
from .donor_calibrated import annotation_peaks, example_hit_matrices, integer_contrast, solve_stratum, verify_mixture


def archived_schedules(context_name, names, inputs, metadata, keypoint_hit, baseline):
    """Score each archived schedule before imposing the annotation-point constraint."""
    context = read_arrays(inputs / f"{context_name}.npz")
    blocks = checked_strata(context, metadata)
    group, donors = context["group"], context["donors"]
    weights = 1 / (len(np.unique(group)) * np.bincount(group)[group])
    peaks = {name: read_arrays(inputs / f"{name}.npz")["peaks"] for name in names}
    anchor = annotation_peaks(context, inputs, context_name)
    window = None
    if metadata is not None:
        side = int(context["side"])
        window = np.unpackbits(context["windows_packed"], axis=1, count=side**2).astype(bool)
    scores = {name: np.zeros(len(donors)) for name in names}
    anchor_gains = np.zeros(len(donors))
    exactly_calibrated = np.ones(len(donors), dtype=bool)
    for block in blocks:
        row, weight = np.arange(len(block)), weights[block[0]]
        relative = np.searchsorted(block, donors[:, block])
        np.testing.assert_array_equal(block[relative], donors[:, block])
        anchor_hits = hit_matrix(anchor, block, context, window, keypoint_hit)[0]
        own_anchor = np.trace(anchor_hits)
        donor_hits = anchor_hits[row, relative].sum(axis=1)
        exactly_calibrated &= (len(block) - 1) * donor_hits == anchor_hits.sum() - own_anchor
        anchor_gains += weight * (own_anchor - donor_hits)
        for name, peak in peaks.items():
            cost, scale = integer_contrast(hit_matrix(peak, block, context, window, keypoint_hit))
            scores[name] += weight * (np.trace(cost) - cost[row, relative].sum(axis=1)) / scale
    for name, draws in scores.items():
        published = baseline["cases"][name]
        np.testing.assert_allclose(draws.mean(), published["published"]["mean"], atol=EPS, rtol=0)
        np.testing.assert_allclose(
            [draws.min(), draws.max()], published["historical_single_schedules"]["range"], atol=EPS, rtol=0,
        )
    return {
        "schedule_count": len(donors), "delta_G": {name: draws.tolist() for name, draws in scores.items()},
        "annotation_gain": anchor_gains.tolist(), "exact_all_strata_calibration_count": int(exactly_calibrated.sum()),
    }


def exact_species_total(matrix, species):
    """Sum each same-species uniform expected hit count as a rational number."""
    total = Fraction()
    for label in np.unique(species):
        members = np.flatnonzero(species == label)
        restricted = matrix[np.ix_(members, members)]
        total += Fraction(int(restricted.sum() - np.trace(restricted)), len(members) - 1)
    return total


def verify_restricted_solver():
    """Check both classes with an independent LP over all six-query permutations."""
    row = np.arange(6)
    species = np.repeat([0, 1], 3)
    allowed = species[:, None] == species
    rng = np.random.default_rng(2026112607)
    all_permutations = np.array([p for p in permutations(row) if np.all(p != row)])
    maximum_error = 0.0
    for _ in range(8):
        cost = rng.integers(-5, 6, size=(6, 6))
        anchor = rng.integers(0, 2, size=(6, 6))
        target = exact_species_total(anchor, species)
        for mask in (np.ones((6, 6), dtype=bool), allowed):
            feasible = all_permutations[mask[row, all_permutations].all(axis=1)]
            values = cost[row, feasible].sum(axis=1)
            constraints = np.stack((np.ones(len(feasible)), anchor[row, feasible].sum(axis=1)))
            reference = linprog(-values, A_eq=constraints, b_eq=[1, float(target)], method="highs")
            result = solve_stratum(cost, anchor, allowed=mask, anchor_target=target)
            verify_mixture(result, cost, anchor, allowed=mask, anchor_target=target)
            if not reference.success:
                raise AssertionError(reference.message)
            maximum_error = max(maximum_error, abs(result["mean"] + reference.fun))
    np.testing.assert_allclose(maximum_error, 0, atol=EPS, rtol=0)
    return {"cases": 16, "maximum_objective_error": maximum_error}


def species_sensitivity(repo, output, keypoint_hit):
    """Compare nested assignment classes on identical queries and anchor targets."""
    data = repo / "data"
    inputs = data / "core_audit/data"
    context_name, case = "new_CUB_context", "new_CUB_native_LocCBM"
    context = read_arrays(inputs / f"{context_name}.npz")
    peaks = read_arrays(inputs / f"{case}.npz")["peaks"]
    anchor = annotation_peaks(context, inputs, context_name)
    species = read_arrays(data / "extensions/donor_species.npz")["species"][context["image"]]
    witness_path = output / "calibrated_witnesses.npz"
    witness = read_arrays(witness_path)
    donor = witness[f"{context_name}__{case}__permutations"][1]
    mixture = witness[f"{context_name}__{case}__mixture"][1]
    blocks = checked_strata(context)
    cross_uniform, cross_witness = [], []
    cross_archived = np.zeros(len(context["donors"]))
    nominal, anchor_gain = [], []
    bounds = {"all_species": [], "same_species": []}
    retained = []
    for index, block in enumerate(blocks):
        same = species[block, None] == species[block]
        same_counts = same.sum(axis=1)
        cross_uniform.append((len(block) - same_counts).mean() / (len(block) - 1))
        a, b = (species[d] != species[block] for d in donor[:, block])
        cross_witness.append(((1 - mixture[index]) * a + mixture[index] * b).mean())
        cross_archived += (species[context["donors"][:, block]] != species[block]).mean(axis=1)

        supported = block[same_counts >= 2]
        if not len(supported):
            raise AssertionError("A frozen attribute has no same-species donor support.")
        retained.extend(supported.tolist())
        labels = species[supported]
        allowed = labels[:, None] == labels
        anch = hit_matrix(anchor, supported, context, None, keypoint_hit)[0].astype(np.int64)
        cost, scale = integer_contrast(hit_matrix(peaks, supported, context, None, keypoint_hit))
        target = exact_species_total(anch, labels)
        uniform = exact_species_total(cost, labels)
        own = int(np.trace(cost))
        normalizer = scale * len(supported)
        nominal.append(float((own - uniform) / normalizer))
        anchor_gain.append(float((int(np.trace(anch)) - target) / len(supported)))
        for rule, mask in (("all_species", None), ("same_species", allowed)):
            endpoints = []
            for sign in (1, -1):
                objective = sign * cost
                solution = solve_stratum(objective, anch, allowed=mask, anchor_target=target)
                verify_mixture(solution, objective, anch, allowed=mask, anchor_target=target)
                endpoints.append((own - sign * solution["mean"]) / normalizer)
            bounds[rule].append(endpoints)
        broad, narrow = bounds["all_species"][-1], bounds["same_species"][-1]
        levels = np.array([broad[0], narrow[0], nominal[-1], narrow[1], broad[1]])
        if np.any(np.diff(levels) < -EPS):
            raise AssertionError("Ranges must be nested and contain the shared baseline.")

    d0 = float(np.mean(nominal))
    report = {}
    for rule, intervals in bounds.items():
        lower, upper = np.mean(intervals, axis=0).tolist()
        share = d0 / (d0 - lower) if d0 > 0 and lower <= 0 else None
        report[rule] = {"range": [lower, upper], "positive_baseline_zero_mixing_share": share}
    return {
        "cross_species_share_full_cohort": {
            "uniform": float(np.mean(cross_uniform)),
            "one_lower_endpoint_witness": float(np.mean(cross_witness)),
            "archived_range": (cross_archived / len(blocks))[[int(cross_archived.argmin()),
                                                              int(cross_archived.argmax())]].tolist(),
        },
        "same_support": {
            "queries": len(retained), "full_queries": len(species), "attributes": len(blocks),
            "images": int(np.unique(context["image"][retained]).size), "full_images": int(np.unique(context["image"]).size),
            "shared_baseline": "Uniform same-species non-self donors. Each attribute weight is equal.",
            "uniform_excess": d0, "annotation_gain": float(np.mean(anchor_gain)), "classes": report,
        },
        "restricted_solver_validation": verify_restricted_solver(),
        "witness_sha256": fingerprint(witness_path),
        "species_sha256": fingerprint(data / "extensions/donor_species.npz"),
    }


def counterexample():
    anchor, matrices = example_hit_matrices()
    row = np.arange(len(anchor))
    witnesses = np.array([[1, 2, 3, 4, 5, 0], [2, 0, 1, 5, 3, 4]])
    np.testing.assert_array_equal(np.sort(witnesses, axis=1), np.broadcast_to(row, witnesses.shape))
    np.testing.assert_array_equal(anchor[row, witnesses].sum(axis=1), [1, 1])
    np.testing.assert_array_equal(matrices["sensitive"][row, witnesses].sum(axis=1), [6, 0])
    return {"anchor": anchor.tolist(), "trained_hits": {name: m.tolist() for name, m in matrices.items()},
            "sensitive_lower_then_upper_donors": witnesses.tolist()}


def main():
    repo, output = options(__doc__)
    sys.path.insert(0, str(repo))
    from audit.io import verify_inputs
    from audit.metrics import keypoint_hit

    verify_inputs(repo / "data")
    inputs = repo / "data/core_audit/data"
    metadata = read_arrays(repo / "experiments/renderer_validity/data/inputs.npz")
    baseline_path = output / "results.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["cohorts"]
    schedules = {}
    for name, cases in CASES.items():
        schedules[name] = archived_schedules(
            name, cases, inputs, metadata if name.startswith("FB") else None, keypoint_hit, baseline[name],
        )
        print(f"Recovered {schedules[name]['schedule_count']} schedules for {name}.", flush=True)
    species = species_sensitivity(repo, output, keypoint_hit)
    write_json(output / "donor_policy_details.json", {
        "baseline_sha256": fingerprint(baseline_path), "schedules": schedules, "species": species,
        "counterexample": counterexample(),
    })
    print(json.dumps(species["same_support"], indent=2))


if __name__ == "__main__":
    main()
