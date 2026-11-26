"""Cross image ownership with matched trained/random concept heads."""
import numpy as np

from .io import check_expected, load_arrays, read_json
from .metrics import keypoint_hit


def score_case(case, root, *, report_head_gap=False):
    """Score one frozen query population and its paired bootstrap draws."""
    arrays = load_arrays(root / case['shared'])
    arrays.update(load_arrays(root / case['file']))
    peaks, donors = arrays['peaks'], arrays['donors']
    images, groups = arrays['image'], arrays['group']
    weights = arrays['weights']
    group_count = int(groups.max() + 1)
    counts = np.bincount(groups, minlength=group_count)
    if (counts == 0).any():
        raise ValueError('A declared macro group has no eligible query')
    observed = np.zeros((weights.shape[1], group_count))
    observed[images, groups] = 1
    denominator = weights @ observed
    if (denominator <= 0).any():
        raise ValueError('Bootstrap draw does not retain all declared groups')
    query_weights = 1 / (group_count * counts[groups])

    if 'windows_packed' in arrays:
        side = int(arrays['side'])
        windows = np.unpackbits(arrays['windows_packed'], axis=1, count=side * side).astype(bool)

        def own_hit(peak):
            # FunnyBirds peak coordinates are (y, x).
            return windows[np.arange(len(images)), peak[:, 0] * side + peak[:, 1]]

        def donor_hit(peak):
            transferred = peak[donors]
            return windows[np.arange(len(images))[None, :],
                           transferred[..., 0] * side + transferred[..., 1]].mean(0)
    else:
        def own_hit(peak):
            return keypoint_hit(peak, arrays['points_xy'])

        def donor_hit(peak):
            return keypoint_hit(peak[donors], arrays['points_xy'][None]).mean(0)

    scores, gains, bootstrap, own_bootstrap = [], [], [], []
    for head_peaks in peaks:
        own = own_hit(head_peaks).astype(float)
        gain = own - donor_hit(head_peaks)
        matrix = np.zeros_like(observed)
        if report_head_gap:
            matrix[images, groups] = own
            own_bootstrap.append(((weights @ matrix) / denominator).mean(1))
        matrix[images, groups] = gain
        scores.append(float(query_weights @ own))
        gains.append(gain)
        bootstrap.append(((weights @ matrix) / denominator).mean(1))
    gains, bootstrap = np.array(gains), np.array(bootstrap)
    point = gains @ query_weights
    result = {'S': scores[0], 'G': float(point[0]),
              'G_ci95': np.quantile(bootstrap[0], [.025, .975]).tolist()}
    if len(peaks) > 1:
        reference = point[1:].mean()
        conditional = bootstrap[0] - bootstrap[1:].mean(0)
        result.update(G_random=float(reference), delta_G=float(point[0] - reference),
                      fixed_head_image_ci95=np.quantile(conditional, [.025, .975]).tolist())
        if case.get('joint', False):
            indices = np.arange(len(weights))[:, None]
            head_indices = arrays['head_indices']
            joint = bootstrap[0] - bootstrap[1:].T[indices, head_indices].mean(1)
            result['image_head_ci95'] = np.quantile(joint, [.025, .975]).tolist()
            if report_head_gap:
                own_bootstrap = np.asarray(own_bootstrap)
                joint_gap = own_bootstrap[0] - own_bootstrap[1:].T[indices, head_indices].mean(1)
                result['head_gap_own'] = float(scores[0] - np.mean(scores[1:]))
                result['head_gap_own_ci95'] = np.quantile(joint_gap, [.025, .975]).tolist()
    return result, bootstrap[0]


def run(data_root):
    root = data_root / 'core_audit'
    protocol = read_json(root / 'manifest.json')
    actual, draws = {}, {}
    for case in protocol['cases']:
        actual[case['name']], draws[case['name']] = score_case(case, root)
    for comparison in protocol['paired_comparisons']:
        first, second = comparison['first'], comparison['second']
        actual[comparison['name']] = {
            'gain_difference': actual[first]['G'] - actual[second]['G'],
            'paired_image_ci95': np.quantile(draws[first] - draws[second], [.025, .975]).tolist()}
    error = check_expected(actual, read_json(root / 'expected_results.json'))
    return {'status': 'passed', 'score_cases': len(protocol['cases']),
            'paired_identity_contrasts': len(protocol['paired_comparisons']),
            'maximum_error': error, 'cases': actual}
