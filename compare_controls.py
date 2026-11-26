"""Compare individual score contrasts and their interaction on the primary cases."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="The public audit repository's frozen data directory")
    parser.add_argument("--output", type=Path, help="Write the comparison to a JSON file")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    from audit.core import score_case
    from audit.io import check_expected, read_json, verify_inputs, write_json

    verify_inputs(args.data_dir)
    root = args.data_dir / "core_audit"
    manifest = read_json(root / "manifest.json")
    cases = {case["name"]: case for case in manifest["cases"]}
    cem = "new_CUB_released_CEM"
    cases[cem] = {
        "name": cem,
        "shared": "data/new_CUB_context.npz",
        "file": "data/new_CUB_released_CEM.npz",
        "joint": True,
    }
    expected = read_json(root / "expected_results.json")
    expected[cem] = read_json(args.data_dir.parent / "docs/released_cem_results.json")
    # These are the primary Gaussian-reference displays in Tables 3 and 4,
    # including the published CEM. Geometry, donors and population stay fixed.
    names = (
        "FB_mask_cam", "FB_mask_gradient",
        "old_CUB_cam", "old_CUB_gradient", "old_CUB_LocCBM",
        "new_CUB_adapted_cam", "new_CUB_adapted_gradient", "new_CUB_native_LocCBM", cem,
    )
    rows = {}
    for name in names:
        scores, _ = score_case(cases[name], root, report_head_gap=True)
        fields = ("S", "G", "G_ci95", "G_random", "delta_G", "image_head_ci95")
        check_expected(scores, {key: expected[name][key] for key in fields})
        rows[name] = {
            "image_pairing": {"estimate": scores["G"], "ci95": scores["G_ci95"]},
            "own_head_gap": {"estimate": scores["head_gap_own"], "ci95": scores["head_gap_own_ci95"]},
            "crossed_excess": {"estimate": scores["delta_G"], "ci95": scores["image_head_ci95"]},
        }

    two_positive = [name for name, row in rows.items()
                    if row["image_pairing"]["ci95"][0] > 0 and row["own_head_gap"]["ci95"][0] > 0]
    interaction_positive = [name for name in two_positive if rows[name]["crossed_excess"]["ci95"][0] > 0]
    result = {
        "scope": "Descriptive comparison with pointwise intervals on overlapping model and cohort cases",
        "case_count": len(rows),
        "positive_individual_contrasts": two_positive,
        "positive_individual_and_crossed_contrasts": interaction_positive,
        "cases": rows,
    }
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
