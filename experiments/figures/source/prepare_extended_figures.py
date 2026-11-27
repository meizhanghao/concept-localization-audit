"""Extract figure inputs from the audit caches without running model inference."""

from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .adapter_sensitivity import grid
from .common import DATA, OUT, ROOT, configure
from .controlled_localization import fields
from .cub_frozen_replication import geometry
from .frozen_adapter_review import hit
from .models import load_meta
from .figure_bundle.render import BASE, sha256


FROZEN = OUT / "cub_frozen_replication_2026-11-20"
CONCEPT = OUT / "concept_axis_2026-11-22"
NATIVE_MAPS = OUT / "frozen_adapter_review_2026-11-21/native_maps_before_projection.npz"
FB_MASKS = OUT / "funnybirds_2026-11-15/test_masks.npy"
POLICY = OUT / "donor_assignment_2026-11-26/query_calibration.json"
MIN_PEAK_SEPARATION = 16


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _window(points):
    y, x = np.mgrid[:224, :224]
    return hit(np.stack((x, y), axis=-1), points)


def select_example(pop, peaks, zeros, scores, pairs):
    """Take the first co-present both-hit case with two legible cached peaks."""
    lengths = [len(pair["query_indices"]) for pair in pairs]
    np.testing.assert_array_equal(scores["pair_id"], np.repeat(np.arange(len(pairs)), lengths))
    queries = np.concatenate([pair["query_indices"] for pair in pairs])
    images = pop["image"][queries]
    np.testing.assert_array_equal(images, scores["image"])
    ids = pop["original_attribute_ids"]
    target = np.repeat([pair["target_attribute"] for pair in pairs], lengths)
    foil = np.repeat([pair["foil_attribute"] for pair in pairs], lengths)
    np.testing.assert_array_equal(target, ids[pop["attribute"][queries]])
    look_up = {
        (int(i), int(ids[c])): q
        for q, (i, c) in enumerate(zip(pop["image"], pop["attribute"]))
    }
    paired_peaks = np.stack((peaks[images, target], peaks[images, foil]), axis=1)
    separation = np.linalg.norm(paired_peaks[:, 0] - paired_peaks[:, 1], axis=1)
    for record in np.flatnonzero((scores["own"][0] == 1).all(0)
                                 & ~zeros[images, target] & ~zeros[images, foil]
                                 & (separation >= MIN_PEAK_SEPARATION)):
        query = int(queries[record])
        other = look_up.get((int(images[record]), int(foil[record])))
        if other is not None and np.array_equal(pop["points_xy"][query], pop["points_xy"][other]):
            return int(record), [query, other], [int(target[record]), int(foil[record])]
    raise ValueError("No eligible co-present example with both cached maps")


def main():
    configure(0)
    files = {
        "population": FROZEN / "population.npz",
        "original_provenance": BASE / "data/provenance.json",
        "original_maps": BASE / "data/maps.npz",
        "funnybirds_masks": FB_MASKS,
        "presence_protocol": CONCEPT / "presence_matched_protocol.json",
        "presence_scores": CONCEPT / "presence_matched_scores.npz",
        "native_peaks": CONCEPT / "all_concept_peaks.npz",
        "native_maps": NATIVE_MAPS,
        "donor_bridge": CONCEPT / "donor_bridge_results.json",
        "donor_calibration": POLICY,
    }
    cases = read(files["original_provenance"])["cases"]
    with np.load(files["population"], allow_pickle=False) as archive:
        pop = {key: archive[key] for key in
               ("points_xy", "image", "image_ids", "attribute", "original_attribute_ids", "rows", "slot")}
    with np.load(files["native_peaks"], allow_pickle=False) as archive:
        cached_peaks, zeros = archive["head0"], archive["zero0"]
    with np.load(files["presence_scores"], allow_pickle=False) as scores:
        pairs = read(files["presence_protocol"])["pairs"]
        record, queries, attributes = select_example(pop, cached_peaks, zeros, scores, pairs)
        own_hits = scores["own"][0, :, record].astype(int)
        pair_index = int(scores["pair_id"][record])

    cub_points = pop["points_xy"][0]
    if int(pop["image_ids"][pop["image"][0]]) != cases[0]["recipient_image_id"]:
        raise ValueError("CUB geometry and RGB provenance do not describe the same image")
    fb_masks = np.load(FB_MASKS, allow_pickle=False, mmap_mode="r")
    fb_mask = fb_masks[cases[1]["recipient_test_index"], 0].astype(bool)
    distance, _ = fields(fb_mask[None, None])
    with np.load(files["original_maps"], allow_pickle=False) as original:
        np.testing.assert_array_equal(original["cub__window"], _window(cub_points))
        np.testing.assert_array_equal(original["fb__window"], distance[0, 0, 0] <= 24)

    image = int(pop["image"][queries[0]])
    image_id = int(pop["image_ids"][image])
    row = int(pop["rows"][image])
    points = pop["points_xy"][queries[0]]
    expected_peaks = cached_peaks[image, attributes]
    meta = load_meta()
    if not ((meta["attrs"][row, attributes] >= .5) & (meta["certainty"][row, attributes] >= 3)).all():
        raise ValueError("Both attributes must be annotated present")
    ctx = geometry(meta, pop["rows"][image:image + 1])
    with np.load(NATIVE_MAPS, allow_pickle=False) as raw:
        maps = torch.as_tensor(raw["head0"][image, pop["slot"][queries], None], device="cuda")
    projected = F.grid_sample(maps, grid(ctx, "own_box").expand(2, -1, -1, -1),
                              mode="bilinear", padding_mode="zeros", align_corners=False)
    maps224 = F.interpolate(projected, size=(224, 224), mode="bicubic", align_corners=False)
    maps224 = maps224[:, 0].cpu().numpy()
    index = maps224.reshape(2, -1).argmax(1)
    peaks = np.stack((index % 224, index // 224), axis=-1)
    np.testing.assert_array_equal(peaks, expected_peaks)
    np.testing.assert_array_equal(hit(peaks, points), own_hits)

    box = ctx["bbox"][image_id]
    name = str(meta["paths"][row])
    with Image.open(DATA / "images" / name) as image_file:
        rgb = np.asarray(image_file.convert("RGB").crop(tuple(box)).resize(
            (224, 224), Image.Resampling.BILINEAR))
    labels = [line.split(maxsplit=1)[1] for line in
              (DATA / "attributes/attributes.txt").read_text().splitlines()]
    record_info = {
        "dataset": "CUB",
        "recipient_image_id": image_id,
        "source_paths": [name],
        "crop_boxes_xyxy": [list(box)],
        "resized_rgb_sha256": [hashlib.sha256(rgb.tobytes()).hexdigest()],
        "selection": f"First stored qualifying record with two nonzero cached hits, "
                     f"shared keypoints, both raw maps and peaks at least {MIN_PEAK_SEPARATION} pixels apart.",
        "pair_index": pair_index,
        "pair_image_record": record,
        "frozen_query_indices": queries,
        "attribute_ids_0based": attributes,
        "concepts": [labels[c] for c in attributes],
        "certainty_ids": meta["certainty"][row, attributes].astype(int).tolist(),
        "peaks_xy": peaks.tolist(),
        "hits": own_hits.tolist(),
    }
    public_directory = ROOT / "github_cun/main"
    public_files = {
        "donor_calibration": public_directory / "experiments/donor_assignment/query_calibration_results.json",
        "donor_bridge": public_directory / "data/donor_bridge/expected.json",
    }
    for key, public_path in public_files.items():
        if read(files[key]) != read(public_path):
            raise ValueError(f"Public results differ from canonical results: {key}")
    if pairs != read(public_directory / "data/donor_bridge/pairs.json"):
        raise ValueError("Public pair records differ from canonical records")
    provenance = {
        "copresent_case": record_info,
        "bridge_pairs": len(pairs),
        "bridge_recipient_records": sum(len(pair["query_indices"]) for pair in pairs),
        "geometry_description": "The same recipient queries and peak caches as the four-cell figure.",
        "author_inputs": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for key, path in files.items()
        },
        "public_results_sha256": {key: sha256(path) for key, path in public_files.items()},
    }
    np.savez_compressed(BASE / "data/extended_examples.npz", cub_annotation_xy=cub_points,
                        fb_part_mask=fb_mask, copresent_annotation_xy=points,
                        copresent_maps=maps224)
    (BASE / "data/extended_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record_info))


if __name__ == "__main__":
    main()
