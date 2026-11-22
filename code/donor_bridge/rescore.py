"""Independently rescore the CUB fixed-recipient donor-rule bridge."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import json
from pathlib import Path
import numpy as np


def hit(peaks, points):
    low = np.clip(peaks[..., None, :] - 45, 0, 224)
    high = np.clip(peaks[..., None, :] + 45, 0, 224)
    return ((points >= low) & (points <= high)).all(-1).any(-1)


def main():
    root = Path(__file__).resolve().parent
    concept = root.parent / 'concept_eligibility'
    geometry = np.load(concept / 'query_geometry.npz')
    data = np.load(concept / 'data.npz')
    peaks = np.load(concept / 'native_peak_cache.npz')['head0']
    pairs = json.loads((root / 'pairs.json').read_text(encoding='utf-8'))
    original_ids = geometry['original_attribute_ids']
    image, attribute, points = geometry['image'], geometry['attribute'], geometry['points_xy']
    gains = [[], []]
    joined_images, joined_pairs = [], []
    for pair_id, pair in enumerate(pairs):
        target, foil = pair['target_attribute'], pair['foil_attribute']
        recipients = np.asarray(pair['query_indices'])
        target_index = int(np.flatnonzero(original_ids == target)[0])
        pools = [np.flatnonzero(attribute == target_index), recipients]
        for rule, pool in enumerate(pools):
            assert len(np.unique(image[pool])) == len(pool)
            assert set(recipients).issubset(set(pool))
            contrast = []
            for label in (target, foil):
                own = hit(peaks[image[recipients], label], points[recipients]).astype(float)
                all_hits = hit(peaks[image[pool], label][None], points[recipients, None]).astype(float)
                donor = (all_hits.sum(1) - own) / (len(pool) - 1)
                contrast.append(own - donor)
            gains[rule].append(contrast[0] - contrast[1])
        joined_images.extend(image[recipients])
        joined_pairs.extend([pair_id] * len(recipients))
    np.testing.assert_array_equal(joined_images, data['pair_image'])
    np.testing.assert_array_equal(joined_pairs, data['pair_id'])
    asymmetric, symmetric = (np.concatenate(v) for v in gains)
    cached = data['pair_own'][0] - data['pair_donor'][0]
    np.testing.assert_allclose(symmetric, cached[0] - cached[1], atol=1e-12, rtol=0)
    pair_ids, images, macro_weights = data['pair_id'], data['pair_image'], data['pair_weights']
    def point(value):
        return float(sum(macro_weights[j] * value[pair_ids == j].mean() for j in range(len(pairs))))
    count = np.zeros((800, len(pairs)))
    np.add.at(count, (images, pair_ids), 1)
    denominator = data['weights'] @ count
    assert (denominator > 0).all()
    difference = symmetric - asymmetric
    total = np.zeros_like(count)
    np.add.at(total, (images, pair_ids), difference)
    draws = (((data['weights'] @ total) / denominator) * macro_weights).sum(1)
    actual = {'asymmetric_I': point(asymmetric), 'symmetric_I': point(symmetric),
              'donor_rule_difference': point(difference),
              'paired_image_ci95': np.quantile(draws, [.025, .975]).tolist()}
    expected = json.loads((root / 'expected.json').read_text(encoding='utf-8'))
    errors = {name: float(np.max(np.abs(np.asarray(value) - expected[name])))
              for name, value in actual.items()}
    assert max(errors.values()) < 1e-12, errors
    actual.update(status='passed', maximum_error=max(errors.values()),
                  dataset='CUB-200-2011', fixed_ordered_pairs=len(pairs),
                  fixed_recipient_observations=len(images))
    (root / 'results.json').write_text(json.dumps(actual, indent=2), encoding='utf-8')
    print(json.dumps(actual, indent=2))


if __name__ == '__main__':
    main()
