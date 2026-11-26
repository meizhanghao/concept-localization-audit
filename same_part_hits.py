"""Describe the scores of co-present Loc-CBM attributes on the frozen CUB queries."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_arrays(path, *names):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[name] for name in names)


def summarize(geometry, pairs, scores, peak_file, score_prefix=""):
    image, attribute, attribute_ids = read_arrays(
        geometry, "image", "attribute", "original_attribute_ids"
    )
    if score_prefix == "pair_":
        image_field = "pair_image"
    else:
        image_field = "image"
    own_bank, donor_bank, images, pair_id, pair_weights = read_arrays(
        scores, f"{score_prefix}own", f"{score_prefix}donor",
        image_field, "pair_id", "pair_weights",
    )
    peaks, = read_arrays(peak_file, "head0")
    counts = np.asarray([len(pair["query_indices"]) for pair in pairs])
    if not len(counts) or (counts < 2).any():
        raise ValueError("Every concept pair needs recipients and at least one donor")
    queries = np.concatenate([pair["query_indices"] for pair in pairs])
    target = np.repeat([pair["target_attribute"] for pair in pairs], counts)
    foil = np.repeat([pair["foil_attribute"] for pair in pairs], counts)
    np.testing.assert_array_equal(pair_id, np.repeat(np.arange(len(pairs)), counts))
    np.testing.assert_array_equal(images, image[queries])
    np.testing.assert_array_equal(target, attribute_ids[attribute[queries]])
    if (target == foil).any():
        raise ValueError("The target and foil must be different attributes")

    own, donor = own_bank[0], donor_bank[0]
    if not np.isin(own, (0, 1)).all():
        raise ValueError("Own scores must be binary hits")
    target_hit, foil_hit = own.astype(bool)
    different_peaks = (peaks[images, target] != peaks[images, foil]).any(axis=1)
    # The paper averages within pair, then within target, then over targets.
    weights = pair_weights[pair_id] / counts[pair_id]
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-12)

    def mean(values):
        return float(weights @ values)

    outcomes = {
        "both_hit": mean(target_hit & foil_hit),
        "both_miss": mean(~target_hit & ~foil_hit),
        "target_only": mean(target_hit & ~foil_hit),
        "foil_only": mean(~target_hit & foil_hit),
    }
    own_contrast = mean(own[0] - own[1])
    donor_contrast = mean(donor[0] - donor[1])
    np.testing.assert_allclose(sum(outcomes.values()), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        own_contrast, outcomes["target_only"] - outcomes["foil_only"],
        rtol=0, atol=1e-12,
    )
    return {
        "observations": len(images),
        "images": len(np.unique(images)),
        "ordered_pairs": len(pairs),
        "target_attributes": len(np.unique(target)),
        "own_hit_outcomes": outcomes,
        "same_own_hit": mean(target_hit == foil_hit),
        "different_own_peaks": mean(different_peaks),
        "different_own_peaks_same_hit": mean(different_peaks & (target_hit == foil_hit)),
        "own_hit_contrast": own_contrast,
        "donor_hit_contrast": donor_contrast,
        "pairing_contrast": own_contrast - donor_contrast,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("data"),
                        help="The public audit repository's data directory")
    inputs.add_argument("--research-cache", type=Path,
                        help="The author's experiment artifacts directory")
    parser.add_argument("--output", type=Path, help="Write the summary to this JSON file")
    args = parser.parse_args()

    if args.research_cache is not None:
        axis = args.research_cache / "concept_axis_2026-11-22"
        cohort = args.research_cache / "cub_frozen_replication_2026-11-20"
        pairs = json.loads((axis / "presence_matched_protocol.json").read_text(encoding="utf-8"))["pairs"]
        result = summarize(cohort / "population.npz", pairs,
                           axis / "presence_matched_scores.npz", axis / "all_concept_peaks.npz")
    else:
        axis = args.data_dir / "concept_eligibility"
        pairs = json.loads((args.data_dir / "donor_bridge/pairs.json").read_text(encoding="utf-8"))
        result = summarize(axis / "query_geometry.npz", pairs, axis / "data.npz",
                           axis / "native_peak_cache.npz", score_prefix="pair_")

    summary = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
    else:
        print(summary, end="")


if __name__ == "__main__":
    main()
