"""Isolate donor eligibility on a fixed CUB recipient/pair population."""
import numpy as np

from .io import check_expected, load_arrays, read_json
from .metrics import keypoint_hit


def run(data_root):
    concept = data_root / 'concept_eligibility'
    root = data_root / 'donor_bridge'
    geometry = load_arrays(concept / 'query_geometry.npz')
    data = load_arrays(concept / 'data.npz')
    peaks = load_arrays(concept / 'native_peak_cache.npz')['head0']
    pairs = read_json(root / 'pairs.json')
    original_ids = geometry['original_attribute_ids']
    images, attribute, points = geometry['image'], geometry['attribute'], geometry['points_xy']
    gains = [[], []]
    joined_images, joined_pairs = [], []
    for pair_id, pair in enumerate(pairs):
        target, foil = pair['target_attribute'], pair['foil_attribute']
        recipients = np.asarray(pair['query_indices'])
        target_index = int(np.flatnonzero(original_ids == target)[0])
        # Enumerate every other eligible image, rather than sampling donors.
        pools = [np.flatnonzero(attribute == target_index), recipients]
        for rule, pool in enumerate(pools):
            if len(np.unique(images[pool])) != len(pool) or not set(recipients).issubset(set(pool)):
                raise ValueError('Recipient/pool membership is inconsistent')
            contrast = []
            for label in (target, foil):
                own = keypoint_hit(peaks[images[recipients], label], points[recipients]).astype(float)
                all_hits = keypoint_hit(peaks[images[pool], label][None], points[recipients, None]).astype(float)
                donor = (all_hits.sum(1) - own) / (len(pool) - 1)
                contrast.append(own - donor)
            gains[rule].append(contrast[0] - contrast[1])
        joined_images.extend(images[recipients])
        joined_pairs.extend([pair_id] * len(recipients))
    np.testing.assert_array_equal(joined_images, data['pair_image'])
    np.testing.assert_array_equal(joined_pairs, data['pair_id'])
    asymmetric, symmetric = (np.concatenate(values) for values in gains)
    cached = data['pair_own'][0] - data['pair_donor'][0]
    np.testing.assert_allclose(symmetric, cached[0] - cached[1], atol=1e-12, rtol=0)
    pair_ids, pair_images, macro_weights = data['pair_id'], data['pair_image'], data['pair_weights']

    def point(values):
        return float(sum(macro_weights[pair] * values[pair_ids == pair].mean()
                         for pair in range(len(pairs))))

    count = np.zeros((data['weights'].shape[1], len(pairs)))
    np.add.at(count, (pair_images, pair_ids), 1)
    denominator = data['weights'] @ count
    if (denominator <= 0).any():
        raise ValueError('Bridge bootstrap draw omits an eligible pair')
    difference = symmetric - asymmetric
    total = np.zeros_like(count)
    np.add.at(total, (pair_images, pair_ids), difference)
    draws = (((data['weights'] @ total) / denominator) * macro_weights).sum(1)
    result = {'asymmetric_I': point(asymmetric), 'symmetric_I': point(symmetric),
              'donor_rule_difference': point(difference),
              'paired_image_ci95': np.quantile(draws, [.025, .975]).tolist()}
    expected = read_json(root / 'expected.json')
    numerical_expected = {name: expected[name] for name in result}
    error = check_expected(result, numerical_expected)
    result.update(status='passed', maximum_error=error, dataset='CUB-200-2011',
                  fixed_ordered_pairs=len(pairs), fixed_recipient_observations=len(pair_images))
    return result
