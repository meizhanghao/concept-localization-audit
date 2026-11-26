"""Audit peak-in-window contrasts over balanced reassignments of cached donors.

Run from main_python with python -B -m concept_null_audit.donor_assignment.
Requires NumPy, SciPy and the existing public audit input bundle.
"""

from itertools import permutations
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

CUB_SIDE = 224
CUB_HALF_WIDTH = 45
EPS = 1e-12
CASES = {
    "FB_mask_context": ("FB_mask_cam", "FB_mask_gradient"),
    "old_CUB_context": ("old_CUB_cam", "old_CUB_gradient", "old_CUB_LocCBM"),
    "new_CUB_context": (
        "new_CUB_adapted_cam", "new_CUB_adapted_gradient",
        "new_CUB_native_LocCBM", "new_CUB_released_CEM",
    ),
}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return dict(arrays)


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def options(description):
    here = Path(__file__).resolve().parent
    root = next((p for p in here.parents if (p / "data/core_audit/manifest.json").is_file()), None)
    canonical = here.name == "concept_null_audit"
    if root is None and canonical:
        root = here.parents[1] / "github_cun/main"
    output = (here / "artifacts/donor_assignment_2026-11-26" if canonical
              else Path("results/donor_assignment"))
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo", type=Path, default=root, required=root is None,
                        help="Root of the public audit checkout with the frozen input bundle.")
    parser.add_argument("--out-dir", type=Path, default=output, help="Write analysis and verification here.")
    args = parser.parse_args()
    return args.repo.resolve(), args.out_dir.resolve()


def freeze_protocol(paths, repo, output):
    protocol = {
        "scope": "Exploratory audit of donor sensitivity on the existing audited cohorts.",
        "cases": CASES,
        "strata": "Exact query and foil within part for FunnyBirds. Same attribute for CUB.",
        "assignment_constraints": "Use each donor once per stratum, exclude self-image.",
        "baseline": "Exact uniform all-other donor mean. Reproduce archived 200-schedule mean separately.",
        "geometry": "Minimize acceptance-window Jaccard distance with a balanced derangement, scoring no maps or heads during matching. Both-empty windows have distance zero.",
        "bounds": "Assign donors to minimize or maximize the trained-minus-mean-random donor hit contrast. Same assignment shared across heads and any compared explainers.",
        "radius": "Minimum global mixture fraction of arbitrary balanced schedules with the uniform schedule that reaches zero. Scope is finite fixed-cohort scores.",
        "intervals": "Paired image and head bootstrap for the fixed assignments only, with the existing draws. Extrema and radius are deterministic finite-cohort summaries.",
        "ranking": "Within each cohort, optimize the difference of CAM and gradient hit contrasts over a common assignment, with fixed head banks.",
        "validation": [
            "Recover every stored baseline score and joint interval to 1e-12.",
            "Exhaustive small derangements agree with both extrema. Check constant-location and annotation controls.",
            "Report an actionable change if a positive exact-uniform interval is no longer positive under the geometry-only assignment, and the paired geometry-minus-uniform interval is below zero.",
        ],
        "inputs": {path.relative_to(repo).as_posix(): fingerprint(path) for path in paths},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    serial = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != serial:
            raise RuntimeError("The frozen donor audit protocol or inputs have changed.")
    else:
        write_json(path, serial)


def matched_assignment(cost):
    """Minimize a square cost matrix with self-donors forbidden."""
    allowed = cost.astype(float, copy=True)
    np.fill_diagonal(allowed, np.inf)
    rows, columns = linear_sum_assignment(allowed)
    if not np.array_equal(rows, np.arange(len(cost))) or not np.isfinite(allowed[rows, columns]).all():
        raise ValueError("No complete eligible derangement.")
    return columns


def jaccard_distance(intersection, area):
    union = area[:, None] + area[None, :] - intersection
    similarity = np.divide(
        intersection, union, out=np.ones(union.shape, dtype=float), where=union > 0,
    )
    return 1 - similarity


def cub_window_geometry(points, side=CUB_SIDE, half_width=CUB_HALF_WIDTH):
    """Exact pairwise window intersection from an at-most-two-square union."""
    if points.shape[1:] != (2, 2):
        raise ValueError("Expected two keypoint slots per CUB query.")
    valid = (points >= 0).all(axis=-1) & (points <= side).all(axis=-1)
    lower = np.clip(points - half_width, 0, side)
    upper = np.clip(points + half_width + 1, 0, side)
    upper = np.where(valid[..., None], upper, lower)
    overlap_lower = lower.max(axis=1)
    overlap_upper = np.maximum(overlap_lower, upper.min(axis=1))
    lower = np.concatenate((lower, overlap_lower[:, None]), axis=1)
    upper = np.concatenate((upper, overlap_upper[:, None]), axis=1)
    signs = (1, 1, -1)
    intersection = np.zeros((len(points), len(points)), dtype=np.int32)
    for left, sign_left in enumerate(signs):
        for right, sign_right in enumerate(signs):
            start = np.maximum(lower[:, left, None], lower[None, :, right])
            end = np.minimum(upper[:, left, None], upper[None, :, right])
            intersection += (sign_left * sign_right * np.maximum(end - start, 0).prod(-1)).astype(np.int32)
    return intersection, np.diag(intersection)


def mask_window_geometry(windows):
    mask = windows.astype(np.float32)
    return mask @ mask.T, mask.sum(axis=1)


def hit_matrix(peaks, block, context, window, keypoint_hit):
    """Rows retain query windows. Columns provide cached peaks in the same stratum."""
    if window is None:
        points = context["points_xy"][block, None]
        return np.array([keypoint_hit(peak[None], points) for peak in peaks[:, block]])
    side = int(context["side"])
    coordinates = peaks[:, block]
    flat = coordinates[..., 0] * side + coordinates[..., 1]
    return window[block[None, :, None], flat[:, None, :]]


def checked_strata(context, metadata=None):
    image, group, donors = (context[key] for key in ("image", "group", "donors"))
    keys = group[:, None]
    if metadata is not None:
        selected = metadata["original200_rows"]
        keys = np.column_stack((
            group, metadata["queries"][selected[image], group],
            metadata["alternatives"][selected[image], group],
        ))
        np.testing.assert_array_equal(
            donors, metadata["original200_donors"][:, image, group] * 5 + group,
        )
    _, codes = np.unique(keys, axis=0, return_inverse=True)
    rows = np.arange(len(image))
    np.testing.assert_array_equal(codes[donors], np.broadcast_to(codes, donors.shape))
    np.testing.assert_array_equal(np.sort(donors, axis=1), np.broadcast_to(rows, donors.shape))
    if np.any(image[donors] == image):
        raise ValueError("Cached schedule contains a self-image donor.")
    blocks = [np.flatnonzero(codes == code) for code in range(codes.max() + 1)]
    if any(len(b) < 2 or len(np.unique(image[b])) != len(b) for b in blocks):
        raise ValueError("Expected at least two distinct images per eligible stratum.")
    return blocks


def balanced_extrema(score, weight):
    rows = np.arange(len(score))
    best_donor = matched_assignment(-score)
    worst_donor = matched_assignment(score)
    return weight * np.array([score[rows, worst_donor].sum(), score[rows, best_donor].sum()])


def sign_radius(baseline, lower, upper):
    """First mixing fraction at which the score can lose its current sign."""
    if abs(baseline) <= EPS:
        return {"status": "zero_at_baseline", "fraction": 0.0}
    endpoint = lower if baseline > 0 else upper
    if baseline * endpoint > 0 and abs(endpoint) > EPS:
        return {"status": "sign_preserved_over_entire_family", "fraction": None}
    return {
        "status": "zero_reachable",
        "fraction": float(np.clip(baseline / (baseline - endpoint), 0, 1)),
    }


def mean_interval(values, weights, groups, image, draws, head_indices):
    """Original query macro average and paired image/head resampling."""
    shape = (draws.shape[1], int(groups.max()) + 1)
    count = np.zeros(shape, dtype=float)
    np.add.at(count, (image, groups), 1)
    denominator = draws @ count
    if (denominator <= 0).any():
        raise ValueError("A resample omitted a query group.")
    summaries = []
    for head in values:
        totals = np.zeros_like(count)
        np.add.at(totals, (image, groups), head)
        summaries.append(((draws @ totals) / denominator).mean(axis=1))
    summaries = np.array(summaries)
    averaged = values @ weights
    paired = summaries[0] - summaries[1:].T[np.arange(len(draws))[:, None], head_indices].mean(axis=1)
    return {
        "mean": float(averaged[0] - averaged[1:].mean()),
        "image_head_ci95": np.quantile(paired, [.025, .975]).tolist(),
    }


def audit_cases(case_names, shared, inputs, groups, metadata, keypoint_hit, expected):
    image, group, donors = (shared[key] for key in ("image", "group", "donors"))
    macro_weight = 1 / (len(np.unique(group)) * np.bincount(group)[group])
    window = None
    if metadata is not None:
        side = int(shared["side"])
        window = np.unpackbits(shared["windows_packed"], axis=1, count=side * side).astype(bool)
    ordered = np.arange(len(image))
    arrays = {name: read_arrays(inputs / f"{name}.npz") for name in case_names}
    if window is not None:
        arrays["FB_mask_layout"] = read_arrays(inputs / "FB_mask_layout.npz")
        arrays["FB_mask_annotation"] = read_arrays(inputs / "FB_mask_annotation.npz")
    schemes = ("published", "uniform", "geometry")
    statistics = {}
    for name, data in arrays.items():
        peaks = data["peaks"]
        side = CUB_SIDE if window is None else int(shared["side"])
        if not ((peaks >= 0) & (peaks < side)).all():
            raise ValueError(f"Peak outside declared grid for {name}.")
        statistics[name] = {
            "gains": {scheme: np.empty(peaks.shape[:2], dtype=float) for scheme in schemes},
            "donor_extrema": np.zeros(2),
            "schedule_scores": np.zeros(len(donors)),
        }
    pair_extrema = np.zeros(2)
    geometry_donor = np.empty(len(image), dtype=int)
    geometry_distances, uniform_distances = [], []

    for block in groups:
        count = len(block)
        local = np.arange(count)
        weight = float(macro_weight[block[0]])
        np.testing.assert_array_equal(macro_weight[block], np.broadcast_to(weight, count))
        relative = np.searchsorted(block, donors[:, block])
        np.testing.assert_array_equal(block[relative], donors[:, block])
        if window is None:
            geometry = cub_window_geometry(shared["points_xy"][block])
        else:
            geometry = mask_window_geometry(window[block])
        distance = jaccard_distance(*geometry)
        nearby = matched_assignment(distance)
        geometry_donor[block] = block[nearby]
        geometry_distances.append(weight * distance[local, nearby].sum())
        uniform_distances.append(weight * (distance.sum() - np.trace(distance)) / (count - 1))
        contrast_costs = {}
        for name, data in arrays.items():
            score = statistics[name]
            hits = hit_matrix(data["peaks"], block, shared, window, keypoint_hit)
            own = hits[:, local, local].astype(float)
            if len(hits) > 1:
                contrast = hits[0] - hits[1:].mean(axis=0)
            else:
                contrast = hits[0].astype(float)
            contrast_costs[name] = contrast
            transferred = hits[:, local, relative]
            score["gains"]["published"][:, block] = own - transferred.mean(axis=1)
            score["gains"]["uniform"][:, block] = own - (hits.sum(axis=-1) - own) / (count - 1)
            score["gains"]["geometry"][:, block] = own - hits[:, local, nearby]
            score["donor_extrema"] += balanced_extrema(contrast, weight)
            score["schedule_scores"] += weight * (np.trace(contrast) - contrast[local, relative].sum(axis=1))
        pair_cost = contrast_costs[case_names[0]] - contrast_costs[case_names[1]]
        pair_extrema += balanced_extrema(pair_cost, weight)

    np.testing.assert_array_equal(np.sort(geometry_donor), ordered)
    if np.any(image[geometry_donor] == image):
        raise ValueError("Geometry matching selected a self-image.")
    results, own_contrasts = {}, {}
    for name, data in arrays.items():
        score = statistics[name]
        gains = score["gains"]
        # Score own-map hits directly to check the archived baseline.
        peak = data["peaks"]
        hits = (keypoint_hit(peak, shared["points_xy"][None]) if window is None
                else window[ordered, peak[..., 0] * side + peak[..., 1]])
        own_by_head = hits @ macro_weight
        own = float(own_by_head[0] - own_by_head[1:].mean() if len(peak) > 1 else own_by_head[0])
        own_contrasts[name] = own
        lower, upper = own - score["donor_extrema"][::-1]
        if not ((score["schedule_scores"] >= lower - EPS) & (score["schedule_scores"] <= upper + EPS)).all():
            raise AssertionError("A realized score lies outside the exact range.")
        if len(peak) > 1:
            def paired(values):
                return mean_interval(values, macro_weight, group, image, shared["weights"], data["head_indices"])

            row = {scheme: paired(gains[scheme]) for scheme in schemes}
            row["geometry_minus_uniform"] = paired(gains["geometry"] - gains["uniform"])
            for key, value in (("S", own_by_head[0]), ("G", gains["published"][0] @ macro_weight),
                               ("G_random", (gains["published"][1:] @ macro_weight).mean()),
                               ("delta_G", row["published"]["mean"]),
                               ("image_head_ci95", row["published"]["image_head_ci95"])):
                np.testing.assert_allclose(value, expected[name][key], atol=EPS, rtol=0)
            row["retires_positive_verdict"] = (
                row["uniform"]["image_head_ci95"][0] > 0
                and row["geometry"]["image_head_ci95"][0] <= 0
                and row["geometry_minus_uniform"]["image_head_ci95"][1] < 0
            )
        else:
            row = {scheme: {"mean": float(gains[scheme][0] @ macro_weight)} for scheme in schemes}
        for scheme in schemes:
            if not lower - EPS <= row[scheme]["mean"] <= upper + EPS:
                raise AssertionError("A summary score lies outside the exact range.")
        row.update(
            own_score=float(own_by_head[0]),
            own_head_gap=own if len(peak) > 1 else None,
            exact_range=[float(lower), float(upper)],
            fraction_to_zero=sign_radius(row["uniform"]["mean"], lower, upper),
            historical_single_schedules={
                "range": [float(score["schedule_scores"].min()), float(score["schedule_scores"].max())],
                "central95": np.quantile(score["schedule_scores"], [.025, .975]).tolist(),
            },
        )
        results[name] = row

    first, second = case_names[:2]
    uniform = results[first]["uniform"]["mean"] - results[second]["uniform"]["mean"]
    own = own_contrasts[first] - own_contrasts[second]
    lower, upper = own - pair_extrema[::-1]
    if window is not None:
        np.testing.assert_allclose(results["FB_mask_layout"]["exact_range"], [0, 0], atol=EPS, rtol=0)
    return {
        "queries": len(image),
        "strata": len(groups),
        "allowed_assignments_edges": sum(len(b) * (len(b) - 1) for b in groups),
        "geometry_only_donor_matching": {
            "uniform_mean_window_distance": float(sum(uniform_distances)),
            "matched_mean_window_distance": float(sum(geometry_distances)),
        },
        "cases": results,
        "common_assignment_cam_minus_gradient": {
            "uniform": uniform, "exact_range": [float(lower), float(upper)],
            "fraction_to_zero": sign_radius(uniform, lower, upper),
        },
    }, geometry_donor


def verify_optimization(keypoint_hit):
    rng = np.random.default_rng(2026112601)
    tested = 0
    for length in range(2, 8):
        rows = np.arange(length)
        choices = np.array([p for p in permutations(rows) if np.all(rows != p)])
        for score in (rng.random((length, length)), rng.integers(-2, 3, (length, length))):
            direct = score[rows, choices].sum(axis=1)
            np.testing.assert_allclose(balanced_extrema(score, 1), [direct.min(), direct.max()], atol=EPS, rtol=0)
            tested += 1
    # Two hit-cost matrices with the same uniform contrast but different sharp ranges.
    robust = np.eye(4) * (2 / 3)
    interchangeable = np.kron(np.eye(2), np.ones((2, 2)))
    examples = []
    for matrix in (robust, interchangeable):
        row = np.arange(len(matrix))
        weight = 1 / len(row)
        own = np.trace(matrix) * weight
        uniform = own - (matrix.sum() - np.trace(matrix)) * weight / (len(row) - 1)
        bounds = own - balanced_extrema(matrix, weight)[::-1]
        np.testing.assert_allclose(uniform, 2 / 3, atol=EPS, rtol=0)
        examples.append({"uniform": float(uniform), "range": bounds.tolist(),
                         "fraction_to_zero": sign_radius(uniform, *bounds)})
    np.testing.assert_allclose(examples[0]["range"], [2/3, 2/3], atol=EPS, rtol=0)
    np.testing.assert_allclose(examples[1]["range"], [0, 1], atol=EPS, rtol=0)
    points = rng.integers(0, 25, size=(30, 2, 2))
    points[:4, 0] = -1
    points[0] = -1
    points[4, 1] = points[4, 0]
    grid = np.stack(np.meshgrid(np.arange(24), np.arange(24)), axis=-1).reshape(-1, 2)
    windows = keypoint_hit(grid[None], points[:, None], half_width=4, side=24)
    for exact, raster in zip(cub_window_geometry(points, 24, 4), mask_window_geometry(windows)):
        np.testing.assert_array_equal(exact, raster)
    return {"exhaustive_derangement_cases": tested, "square_union_matches_grid": True, "toy_cases": examples}


def main():
    # Use the same scorer and input integrity checks as the public audit.
    repo, output = options(__doc__)
    sys.path.insert(0, str(repo))
    from audit.io import read_json, verify_inputs
    from audit.metrics import keypoint_hit

    data = repo / "data"
    verified = verify_inputs(data)
    inputs = data / "core_audit/data"
    expected_path = data / "core_audit/expected_results.json"
    cem_path = repo / "docs/released_cem_results.json"
    metadata_path = repo / "experiments/renderer_validity/data/inputs.npz"
    paths = [metadata_path, expected_path, cem_path]
    for context, names in CASES.items():
        paths.extend(inputs / f"{name}.npz" for name in (context, *names))
    paths += [inputs / f"FB_mask_{name}.npz" for name in ("layout", "annotation")]
    freeze_protocol(paths, repo, output)
    checks = verify_optimization(keypoint_hit)
    expected = read_json(expected_path)
    expected["new_CUB_released_CEM"] = read_json(cem_path)
    results, witnesses = {}, {}
    for context, names in CASES.items():
        shared = read_arrays(inputs / f"{context}.npz")
        metadata = read_arrays(metadata_path) if context.startswith("FB") else None
        strata = checked_strata(shared, metadata)
        summary, geometry_donor = audit_cases(names, shared, inputs, strata, metadata, keypoint_hit, expected)
        results[context] = summary
        witnesses[context] = geometry_donor
        print(f"Verified and scored {context} ({len(geometry_donor):,} queries).", flush=True)
    write_json(output / "results.json", {
        "baseline_inputs_verified": verified, "self_checks": checks, "cohorts": results,
    })
    np.savez_compressed(output / "geometry_assignments.npz", **witnesses)


if __name__ == "__main__":
    main()
