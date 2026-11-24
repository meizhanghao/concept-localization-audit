"""Check released-CEM scores with clipped rectangles and scalar weighted means.

First run audit_released_cem.py. This CPU-only check reconstructs every saved
image/head interval; it does not repeat checkpoint, label or autograd checks.
See docs/released_cem.md for those checks and the raw-image generator.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit.io import load_arrays, read_json, verify_inputs, write_json


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/released_cem')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify_inputs(ROOT / 'data')
    context = load_arrays(ROOT / 'data/core_audit/data/new_CUB_context.npz')
    bank = load_arrays(ROOT / 'data/core_audit/data/new_CUB_released_CEM.npz')
    result = read_json(args.results_dir / 'results.json')
    points, peak = context['points_xy'], bank['peaks']
    valid = ((points >= 0) & (points <= 224)).all(-1)

    def hit(coordinate):
        x, y = coordinate[..., None, 0], coordinate[..., None, 1]
        accepted = (points[..., 0] >= np.maximum(x - 45, 0)) & (points[..., 0] <= np.minimum(x + 45, 224))
        accepted &= (points[..., 1] >= np.maximum(y - 45, 0)) & (points[..., 1] <= np.minimum(y + 45, 224))
        return (accepted & valid).any(-1)

    own = hit(peak).astype(float)
    donor = np.zeros_like(own)
    for schedule in context['donors']:
        donor += hit(peak[:, schedule])
    donor /= len(context['donors'])
    gain = own - donor
    groups = [np.flatnonzero(context['group'] == a) for a in range(40)]

    def macro(value):
        return np.mean([value[..., ix].mean(-1) for ix in groups], axis=0)

    errors = []

    def check(actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        errors.append(float(np.max(np.abs(actual - expected))))

    point = macro(gain)
    check(macro(own)[0], result['S'])
    check(point[0], result['G'])
    check(point[1:], result['individual_reference_G'])
    check(point[1:].mean(), result['G_random'])
    check(point[0] - point[1:].mean(), result['delta_G'])
    for name, value in [('trained_own', macro(own)[0]), ('trained_donor', macro(donor)[0]),
                        ('reference_own', macro(own)[1:].mean()), ('reference_donor', macro(donor)[1:].mean())]:
        check(value, result['cells'][name])

    n_draws = len(context['weights'])
    draws = np.empty((n_draws, 2))
    fixed = np.empty(n_draws)
    conditional = gain[0] - gain[1:].mean(0)
    for at, weight in enumerate(context['weights']):
        w = weight[context['image']]
        contrast = gain[0] - gain[1:][bank['head_indices'][at]].mean(0)
        draws[at, 0] = np.mean([np.average(gain[0, ix], weights=w[ix]) for ix in groups])
        draws[at, 1] = np.mean([np.average(contrast[ix], weights=w[ix]) for ix in groups])
        fixed[at] = np.mean([np.average(conditional[ix], weights=w[ix]) for ix in groups])
    for values, key in [(draws[:, 0], 'G_ci95'), (draws[:, 1], 'image_head_ci95'), (fixed, 'fixed_head_image_ci95')]:
        check(np.quantile(values, [.025, .975]), result[key])
    report = {'status': 'passed', 'maximum_error': max(errors),
        'image_and_head_draws_independently_reconstructed': n_draws,
        'fixed_head_and_joint_intervals_checked': True,
        'geometry': 'Independent clipped peak-centered rectangles, inclusive coordinates.',
        'scope': 'CPU peak-score and interval verification; no checkpoint forward, label reconstruction, autograd or training checks.'}
    write_json(args.output or args.results_dir / 'independent_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
