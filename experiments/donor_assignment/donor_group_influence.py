"""Attribute influence on the fixed-cohort Loc-CBM donor sensitivity bound."""

from itertools import combinations
import json
import sys

import numpy as np

from .donor_assignment import (
    EPS, checked_strata, fingerprint, hit_matrix, options, read_arrays, sign_radius,
    write_json,
)
from .donor_calibrated import annotation_peaks, integer_contrast, solve_stratum, verify_mixture


CONTEXT = "new_CUB_context"
METHOD = "new_CUB_native_LocCBM"


def influence(rows):
    """Drop entire attributes and reaverage the remaining stratum optima."""
    result = {}
    for removed in (1, 2):
        summaries = []
        for omitted in combinations(range(len(rows)), removed):
            remaining = [r for index, r in enumerate(rows) if index not in omitted]
            baseline = float(np.mean([r["uniform_excess"] for r in remaining]))
            lower = float(np.mean([r["lower_bound"] for r in remaining]))
            radius = positive_baseline_mixture(baseline, lower)
            summaries.append({
                "excluded_group_ids": [rows[index]["group_id"] for index in omitted],
                "uniform_excess": baseline,
                "lower_bound": lower,
                "minimum_mixture_share": radius["fraction"],
                "status": radius["status"],
            })
        shares = [r["minimum_mixture_share"] for r in summaries
                  if r["minimum_mixture_share"] is not None]
        result[str(removed)] = {
            "exclusions": len(summaries),
            "positive_baseline_and_negative_lower": sum(
                r["uniform_excess"] > 0 > r["lower_bound"] for r in summaries
            ),
            "mixture_share_range": [min(shares), max(shares)] if shares else None,
            "rows": summaries,
        }
    return result


def positive_baseline_mixture(baseline, lower):
    """Compute the downward zero crossing from a positive uniform excess."""
    if baseline < -EPS:
        return {"status": "baseline_negative", "fraction": None}
    return sign_radius(baseline, lower, baseline)


def main():
    repo, output = options(__doc__)
    sys.path.insert(0, str(repo))
    from audit.io import verify_inputs
    from audit.metrics import keypoint_hit

    verified = verify_inputs(repo / "data")
    inputs = repo / "data/core_audit/data"
    attribute_file = inputs.parent / "attribute_manifest.json"
    context_file, method_file = (inputs / f"{name}.npz" for name in (CONTEXT, METHOD))
    attributes = json.loads(attribute_file.read_text(encoding="utf-8"))["attributes"]
    by_group = {int(a["group_id"]): a for a in attributes}
    shared = read_arrays(context_file)
    trained_and_random = read_arrays(method_file)["peaks"]
    anchor = annotation_peaks(shared, inputs, CONTEXT)
    expected_file = output / "calibrated_results.json"
    expected = json.loads(expected_file.read_text(encoding="utf-8"))["cohorts"][CONTEXT]

    rows = []
    for block in checked_strata(shared):
        group = int(shared["group"][block[0]])
        score, head_count = integer_contrast(
            hit_matrix(trained_and_random, block, shared, None, keypoint_hit)
        )
        annotation = hit_matrix(anchor, block, shared, None, keypoint_hit)[0].astype(np.int64)
        optimum = solve_stratum(score, annotation)
        verify_mixture(optimum, score, annotation)
        n = len(block)
        own = int(np.trace(score))
        uniform = (own - (score.sum() - own) / (n - 1)) / (n * head_count)
        lower = (own - optimum["mean"]) / (n * head_count)
        if lower > uniform + EPS:
            raise AssertionError("Uniform donors must be feasible in every attribute.")
        rows.append({
            "group_id": group,
            "attribute": by_group[group]["name"],
            "part": by_group[group]["part"],
            "queries": n,
            "uniform_excess": float(uniform),
            "lower_bound": float(lower),
            "worst_case_drop": float(uniform - lower),
            "uniform_annotation_gain": float(
                (np.trace(annotation) - (annotation.sum() - np.trace(annotation)) / (n - 1)) / n
            ),
        })
    if set(by_group) != {r["group_id"] for r in rows}:
        raise AssertionError("Attribute membership does not match the frozen manifest.")
    baseline = float(np.mean([r["uniform_excess"] for r in rows]))
    lower = float(np.mean([r["lower_bound"] for r in rows]))
    annotation_gain = float(np.mean([r["uniform_annotation_gain"] for r in rows]))
    radius = positive_baseline_mixture(baseline, lower)
    case = expected["cases"][METHOD]
    np.testing.assert_allclose(
        [baseline, lower, annotation_gain, radius["fraction"]],
        [case["uniform"], case["exact_range"][0], expected["fixed_annotation_gain"],
         case["fraction_to_zero"]["fraction"]],
        rtol=0, atol=EPS,
    )
    total_drop = sum(r["worst_case_drop"] for r in rows)
    for row in rows:
        row["share_of_total_drop"] = row["worst_case_drop"] / total_drop
    largest = sorted(rows, key=lambda row: row["share_of_total_drop"], reverse=True)
    np.testing.assert_allclose(sum(r["share_of_total_drop"] for r in rows), 1, atol=EPS)
    report = {
        "scope": "Exploratory leave-attribute-out sensitivity in the fixed disjoint CUB cohort.",
        "inputs_verified": verified,
        "input_sha256": {p.relative_to(repo).as_posix(): fingerprint(p)
                         for p in (attribute_file, context_file, method_file)},
        "calibrated_result_sha256": fingerprint(expected_file),
        "n_attributes": len(rows),
        "n_queries": sum(r["queries"] for r in rows),
        "uniform_excess": baseline,
        "lower_bound": lower,
        "minimum_mixture_share": radius["fraction"],
        "attributes_with_negative_lower_bound": sum(r["lower_bound"] < 0 for r in rows),
        "largest_attribute_drop_share": largest[0]["share_of_total_drop"],
        "largest_two_attributes_drop_share": sum(r["share_of_total_drop"] for r in largest[:2]),
        "rows": rows,
        "leave_attributes_out": influence(rows),
    }
    write_json(output / "attribute_influence.json", report)
    print(f"Verified {len(rows)} attribute optima and all one- and two-attribute omissions.", flush=True)


if __name__ == "__main__":
    main()
