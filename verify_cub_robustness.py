"""Independently check paired-window and anatomical follow-up arithmetic.

First run analyze_cub_robustness.py. This checker uses scalar attribute means
for selected raw-query draws and reconstructs all saved interval families.
It does not rerun model inference or certify finite-sample interval coverage.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from audit.io import load_arrays, read_json, verify_inputs, write_json


ROOT = Path(__file__).resolve().parent
WIDTHS = (16, 24, 36, 45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/robustness')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify_inputs(ROOT / 'data')
    result = read_json(args.results_dir / 'results.json')
    context = load_arrays(ROOT / 'data/core_audit/data/new_CUB_context.npz')
    bank = load_arrays(ROOT / 'data/core_audit/data/new_CUB_native_LocCBM.npz')
    draws = load_arrays(args.results_dir / 'bootstrap_draws.npz')
    old = read_json(ROOT / 'docs/cub_geometry_results.json')['results']
    errors = []

    def check(actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        errors.append(float(np.max(np.abs(actual - expected))))

    width_draws = draws['width_draws']
    attribute_draws = draws['attribute_delta_draws']
    assert width_draws.shape == (3000, 4) and attribute_draws.shape == (3000, 40)
    for at, half in enumerate(WIDTHS):
        for key in ('S', 'G', 'G_random', 'delta_G'):
            check(result['width_results'][str(half)][key], old[str(half)][key])
        check(np.quantile(width_draws[:, at], [.025, .975]), old[str(half)]['image_head_ci95'])
    pairs = list(itertools.combinations(range(4), 2))
    pair_draws = np.stack([width_draws[:, b] - width_draws[:, a] for a, b in pairs], axis=1)
    estimates = np.array([result['width_results'][str(h)]['delta_G'] for h in WIDTHS])
    pair_estimates = np.array([estimates[b] - estimates[a] for a, b in pairs])
    for at, record in enumerate(result['paired_width_comparisons']):
        a, b = pairs[at]
        assert record['narrower'] == WIDTHS[a] and record['wider'] == WIDTHS[b]
        check(pair_estimates[at], record['delta_difference'])
        check(np.quantile(pair_draws[:, at], [.025, .975]), record['pointwise_ci95'])

    # Rebuild both maximum-standardized-deviation families from saved draws.
    for point, values, stored in [
        (estimates, width_draws, result['four_width_simultaneous_band']),
        (pair_estimates, pair_draws, result['six_pair_band_details']),
    ]:
        scale = np.std(values, axis=0, ddof=1)
        maxima = np.max(np.abs(values - point) / scale, axis=1)
        critical = np.quantile(maxima, .95)
        check(scale, stored['standard_errors'])
        check(critical, stored['critical'])
        check(np.column_stack((point - critical * scale, point + critical * scale)), stored['ci95'])
        check(np.mean(maxima <= critical), stored['empirical_centered_coverage'])
    for at, row in enumerate(result['paired_width_comparisons']):
        check(row['six_comparison_simultaneous_ci95'], result['six_pair_band_details']['ci95'][at])

    check(attribute_draws.mean(1), width_draws[:, -1])
    peak, points = bank['peaks'], context['points_xy']
    valid = ((points >= 0) & (points <= 224)).all(-1)

    def hits(coordinate):
        return ((np.abs(coordinate[..., None, :] - points) <= 45).all(-1) & valid).any(-1)

    own = hits(peak)
    other = np.zeros_like(own, dtype=float)
    for schedule in context['donors']:
        other += hits(peak[:, schedule])
    gain = own.astype(float) - other / len(context['donors'])
    image, group = context['image'], context['group']
    per_attribute = [np.flatnonzero(group == a) for a in range(40)]
    checked_draws = np.linspace(0, len(width_draws) - 1, 80, dtype=int)
    for draw in checked_draws:
        weight = context['weights'][draw, image]
        local = gain[0] - gain[1:][bank['head_indices'][draw]].mean(0)
        scalar = [np.average(local[ix], weights=weight[ix]) for ix in per_attribute]
        check(scalar, attribute_draws[draw])

    assert len(result['paired_width_comparisons']) == 6
    assert len(result['parts']) == 10 and len(result['attributes']) == 40
    covered = []
    for row in result['parts']:
        ids = row['attribute_groups']
        covered.extend(ids)
        rest = np.setdiff1d(np.arange(40), ids)
        check(np.quantile(attribute_draws[:, ids].mean(1), [.025, .975]), row['pointwise_ci95'])
        check(np.quantile(attribute_draws[:, rest].mean(1), [.025, .975]), row['leave_out_pointwise_ci95'])
    assert sorted(covered) == list(range(40))
    check(sum(row['macro_contribution'] for row in result['parts']), estimates[-1])
    report = {'status': 'passed', 'maximum_error': max(errors),
        'old_four_width_results_unchanged': True,
        'raw_scalar_image_head_draws_independently_checked': len(checked_draws),
        'all_3000_width_difference_and_group_interval_draws_checked': True,
        'four_width_and_six_pair_bands_reconstructed': True,
        'all_six_width_pairs_and_ten_parts_reported': True,
        'scope': 'Numerical arithmetic and centered-bootstrap implementation; not model inference or exact finite-sample coverage.'}
    write_json(args.output or args.results_dir / 'independent_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
