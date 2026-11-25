"""Extend the existing training-only layout prior to the frozen CUB audit card.

Prepare freezes the rule before new prior scores; evaluate never retunes it.
This is a useful control, not a novel explanation or independent confirmation.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .common import DATA, OUT, PREVIOUS, configure, write_json
from .models import load_meta
from .identity_controls import family
from .published_metric_gate import FAMILIES, SOURCE, peaks224, scorer_function
from .cub_frozen_replication import geometry
from .cub_prior_statistics import analyze, training_centers, hits

DEST = OUT / 'cub_position_prior_2026-11-25'
FROZEN = OUT / 'cub_frozen_replication_2026-11-20'
OLD = OUT / 'identity_controls_2026-11-16'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inputs_unchanged(protocol):
    expected = dict(protocol['inputs'])
    amendment = DEST / 'implementation_amendment.json'
    if amendment.exists():
        change = read(amendment)
        path = str(Path(__file__))
        assert expected[path] == change['original_source_sha256']
        expected[path] = change['amended_source_sha256']
    return all(digest(p) == h for p, h in expected.items())


def prepare():
    DEST.mkdir(exist_ok=True)
    inputs = [DATA / 'parts/part_locs.txt', DATA / 'bounding_boxes.txt',
              DATA / 'train_test_split.txt', OUT / 'sample_manifest.json',
              PREVIOUS / '可替换部件参照初步验证_2026-11-14/metadata.npz',
              OLD / 'C3_provenance.json', FROZEN / 'population.npz',
              FROZEN / 'query_scores.npz', FROZEN / 'bootstrap_draws.npz', SOURCE,
              Path(__file__), Path(__file__).with_name('cub_prior_statistics.py')]
    protocol = {
        'date': '2026-11-25',
        'status': 'Frozen before new prior outcomes; exploratory follow-up in a known cohort.',
        'prior': 'Reuse historical training-only family means from 4794 split0 images. Point-weighted mean of visible keypoints in each own bird-box normalized frame. No concept-presence/class/correctness filtering. Gaussian56 sigma=.035 at cell centers, then unchanged CUDA bicubic224 peak extraction. Identical peak for every image of a given concept/part.',
        'historical_relation': 'This is the pre-existing C3 training_part_layout56 rule applied to the frozen 800-photo cohort, not a new method or a discovery claim. Verify family means against historical provenance.',
        'population': 'Existing 800 test photos,40 inherited attributes,8323 queries; no new eligibility, filtering, donors, heads, window or model.',
        'annotation_use': 'Training keypoints estimate centers. Test keypoints ONLY score peaks. The shared own-bird-box evaluation frame still uses test box annotations; not an annotation-free image-space prior.',
        'scorer': 'Unchanged ProtoCBM keypoint function at h45. Check all40 prior per-attribute scores against independent NumPy membership.',
        'uncertainty': 'Reuse existing3000 image-cluster weights; fixed training calibration. Primary paired contrast native_LocCBM S minus prior S. Other two model contrasts descriptive, not multiplicity-adjusted.',
        'outputs': ['prior S and G', 'exact uniform-peak chance', 'native S minus prior paired95% interval', 'all3 existing model/prior contrasts', 'per-part centers/counts', 'source/scorer/split consistency checks'],
        'inclusion_gate': 'Integrity and scorer checks must pass. Include if it completes the missing chance/prior/reference card and changes or constrains interpretation of S. No requirement that prior wins or that a paired interval exclude zero. Do not infer semantic/causal validity, universal chance, or independent confirmation.',
        'inputs': {str(p): digest(p) for p in inputs}}
    path = DEST / 'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'Frozen protocol changed; do not silently rerun a new rule'
    else:
        write_json(path, protocol)
    print('Prior protocol frozen before scoring', flush=True)


def evaluate():
    configure(0)
    protocol = read(DEST / 'protocol.json')
    assert inputs_unchanged(protocol)
    meta = load_meta()
    train = np.flatnonzero(meta['splits'] == 0)
    assert len(train) == 4794
    pop = dict(np.load(FROZEN / 'population.npz'))
    scores = np.load(FROZEN / 'query_scores.npz')
    draws = np.load(FROZEN / 'bootstrap_draws.npz')
    assert not np.intersect1d(meta['image_ids'][train], pop['image_ids']).size
    official_train = {int(a): int(b) for a, b in
                      (line.split() for line in (DATA / 'train_test_split.txt').read_text().splitlines())}
    assert all(official_train[int(i)] == 1 for i in meta['image_ids'][train])
    assert all(official_train[int(i)] == 0 for i in pop['image_ids'])
    corner = meta['bbox'][train, :2] - meta['bbox'][train, 2:] / 2
    coords = (meta['part_xy'][train] - corner[:, None]) / meta['bbox'][train, None, 2:]
    names = list(FAMILIES)
    centers, counts = training_centers(coords, meta['part_visible'][train] > .5,
                                       list(FAMILIES.values()))
    historical = read(OLD / 'C3_provenance.json')['training_layout_centers_in_bbox']
    max_center_error = max(float(np.max(np.abs(centers[i] - historical[n])))
                           for i, n in enumerate(names))
    # The historical code accumulated float32 points; summation order is fixed.
    assert max_center_error < 1e-6
    centers = np.asarray([historical[n] for n in names], dtype=np.float32)
    yy, xx = (np.mgrid[:56, :56] + .5) / 56
    maps = np.exp(-((xx[None] - centers[:, 0, None, None]) ** 2 +
                    (yy[None] - centers[:, 1, None, None]) ** 2) / (2 * .035 ** 2)).astype(np.float32)
    peaks = peaks224(maps[:, None], device='cuda')[:, 0]
    manifest = read(OUT / 'sample_manifest.json')
    attribute_part = np.array([names.index(family(manifest['attribute_names'][q]))
                               for q in pop['queries']])
    points = pop['points_xy']
    # Exactly enumerate accepted pixels. This also handles clipping and unions.
    yy, xx = np.mgrid[:224, :224]
    pixels = np.stack([xx, yy], -1)
    chance = np.array([hits(pixels, p).mean() for p in points])
    packed = {
        'training_image_ids': meta['image_ids'][train],
        'training_points_in_box': coords, 'training_visible': meta['part_visible'][train] > .5,
        'target_image_ids': pop['image_ids'], 'image': pop['image'],
        'attribute': pop['attribute'], 'attribute_part': attribute_part,
        'part_peaks_xy': peaks, 'training_centers': centers,
        'points_xy': points, 'donors': pop['donors'],
        'weights': draws['weights'], 'chance': chance,
        **{n + '__S': scores[n + '__head0__S'] for n in
           ['native_LocCBM', 'adapted_cam', 'adapted_gradient']}}
    np.savez_compressed(DEST / 'inputs.npz', **packed)
    with threadpool_limits(limits=2):
        result = analyze(packed)
    # Verify prior centers via independent float64 means to expose any drift.
    independent = []
    for parts in FAMILIES.values():
        selected = [coords[i, p].astype(float) for i in range(len(train))
                    for p in parts if meta['part_visible'][train[i], p] > .5]
        independent.append(np.mean(selected, 0))
    independent_error = float(np.abs(np.asarray(independent) - centers).max())
    assert independent_error < 5e-6
    ctx = geometry(meta, pop['rows'])
    labels = pop['observed'].astype(int)
    env = {'attributes_names': [family(manifest['attribute_names'][q]) for q in pop['queries']],
           'part_attributes_names': names, 'part_name_to_part_indexes': FAMILIES,
           'id_to_bbox': ctx['bbox'], 'id_to_part_loc': ctx['parts'],
           'id_to_attributes': {int(i): labels[j] for j, i in enumerate(ctx['ids'])}}
    scorer = scorer_function(env)
    query_peaks = peaks[attribute_part[pop['attribute']]]
    impulse = torch.zeros((len(query_peaks), 224, 224), device='cuda')
    impulse[torch.arange(len(query_peaks), device='cuda'),
            torch.tensor(query_peaks[:, 1], device='cuda'),
            torch.tensor(query_peaks[:, 0], device='cuda')] = 1
    _, (byattr, _, _) = scorer(
        [impulse[torch.tensor(pop['attribute'] == a, device='cuda')] for a in range(40)],
        torch.tensor(ctx['ids'], device='cuda'), bbox_half_size=45, img_size=224)
    expected = hits(query_peaks, points)
    official_error = float(np.max(np.abs(byattr - np.array(
        [expected[pop['attribute'] == a].mean() for a in range(40)]))))
    assert official_error < 1e-6
    result['prior_centers'] = {n: {'center_xy_in_box': centers[i].tolist(),
                                      'visible_training_points': int(counts[i]),
                                      'scored_peak_xy': peaks[i].tolist()}
                               for i, n in enumerate(names)}
    result['inclusion_decision'] = {
        'useful': True, 'reason': 'Completes the declared position-prior context on the same photo cohort; compare S without assuming a prior must win.',
        'not_a_new_method': True, 'no_new_confirmation_claim': True}
    write_json(DEST / 'results.json', result)
    write_json(DEST / 'verification.json', {
        'status': 'passed', 'train_images': len(train), 'train_test_overlap': 0,
        'official_train_test_split_consistent': True,
        'historical_center_max_error': max_center_error,
        'independent_float64_center_max_error': independent_error,
        'unchanged_official_scorer_max_error': official_error,
        'prior_query_G_exact_zero': result['prior_G']['mean'] == 0,
        'protocol_inputs_unchanged_except_declared_diagnostic_amendment': inputs_unchanged(protocol)})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    args = parser.parse_args()
    globals()[args.stage]()
