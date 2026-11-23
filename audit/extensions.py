"""Recompute reference-family and donor-species sensitivity from frozen peaks."""
import numpy as np

from .io import check_expected, load_arrays, read_json
from .metrics import keypoint_hit
from .new_input import evaluate


def species_comparison(context, bank, species):
    """Enumerate legal donors on metadata-defined common recipient support."""
    image, group = context['image'], context['group']
    species = species[image]
    other_same = np.array([np.sum((group == g) & (species == s)) - 1 for g, s in zip(group, species)])
    keep = other_same >= 1
    selected = np.flatnonzero(keep)
    macro_groups = int(group.max() + 1)
    counts = np.bincount(group[keep], minlength=macro_groups)
    if (counts == 0).any():
        raise ValueError('Species-constrained common support loses a fixed macro group')
    alpha = 1 / (macro_groups * counts[group[keep]])
    peak = bank['peaks']
    own = keypoint_hit(peak[:, keep], context['points_xy'][None, keep]).astype(float)
    donor_scores = {rule: np.zeros_like(own) for rule in ('all_species', 'same_species')}
    for g in range(macro_groups):
        positions = np.flatnonzero(group[selected] == g)
        target = selected[positions]; source = np.flatnonzero(group == g)
        all_legal = image[target, None] != image[source][None]
        same_legal = all_legal & (species[target, None] == species[source][None])
        for h in range(len(peak)):
            hits = keypoint_hit(peak[h, source][None], context['points_xy'][target, None])
            for rule, legal in [('all_species', all_legal), ('same_species', same_legal)]:
                donor_scores[rule][h, positions] = (hits * legal).sum(1) / legal.sum(1)
    observed = np.zeros((context['weights'].shape[1], macro_groups))
    np.add.at(observed, (image[keep], group[keep]), 1)
    denominator = context['weights'] @ observed
    if (denominator <= 0).any():
        raise ValueError('A bootstrap draw loses a macro group on common support')
    def aggregate(values):
        matrix = np.zeros_like(observed)
        np.add.at(matrix, (image[keep], group[keep]), values)
        return ((context['weights'] @ matrix) / denominator).mean(1)
    result = {}; contrasts = {}
    for rule, donor in donor_scores.items():
        gains = own - donor; point = gains @ alpha
        boot = np.array([aggregate(v) for v in gains]).T
        reference = boot[:, 1:][np.arange(len(boot))[:, None], bank['head_indices']].mean(1)
        joint = boot[:, 0] - reference; contrasts[rule] = joint
        result[rule] = {'S': float(own[0] @ alpha), 'G': float(point[0]), 'G_random': float(point[1:].mean()),
                        'delta_G': float(point[0] - point[1:].mean()),
                        'image_head_ci95': np.quantile(joint, [.025, .975]).tolist()}
    difference = contrasts['same_species'] - contrasts['all_species']
    return {'conditions': result,
            'paired_donor_rule_difference': {'mean': result['same_species']['delta_G'] - result['all_species']['delta_G'],
                                           'image_head_ci95': np.quantile(difference, [.025, .975]).tolist()},
            'queries': int(keep.sum()), 'images': int(np.unique(image[keep]).size), 'attributes': macro_groups,
            'species': int(np.unique(species[keep]).size)}


def run(data_root):
    context = load_arrays(data_root / 'core_audit/data/new_CUB_context.npz')
    bank = load_arrays(data_root / 'core_audit/data/new_CUB_native_LocCBM.npz')
    permutation = load_arrays(data_root / 'extensions/reference_family.npz')
    arrays = {'peaks_xy': np.concatenate([bank['peaks'][:1], permutation['peaks']]),
              'image_id': context['image'], 'group_id': context['group'], 'concept_id': context['group'],
              'points_xy': context['points_xy'], 'donors': context['donors'],
              'bootstrap_weights': context['weights'], 'head_indices': permutation['head_indices']}
    config = {'geometry': {'type': 'keypoints', 'side': 224, 'half_width': 45},
              'eligibility': 'Original frozen full8323-query cohort; channel-permutation reference family.'}
    permuted = evaluate(arrays, config)
    # Same core arrays and identical bootstrap weights; independent head draws for
    # the two families are already frozen in the released input arrays.
    def joint_draws(peaks, indices):
        own = keypoint_hit(peaks, context['points_xy'][None]).astype(float)
        donor = keypoint_hit(peaks[:, context['donors']], context['points_xy'][None, None]).mean(1)
        gains = own - donor
        observed = np.zeros((context['weights'].shape[1], 40))
        observed[context['image'], context['group']] = 1
        denominator = context['weights'] @ observed
        boot = []
        for value in gains:
            matrix = np.zeros_like(observed); matrix[context['image'], context['group']] = value
            boot.append(((context['weights'] @ matrix) / denominator).mean(1))
        boot = np.array(boot).T
        return boot[:, 0] - boot[:, 1:][np.arange(len(boot))[:, None], indices].mean(1)
    gaussian_joint = joint_draws(bank['peaks'], bank['head_indices'])
    permutation_joint = joint_draws(arrays['peaks_xy'], permutation['head_indices'])
    reference_result = {k: permuted[k] for k in ['S', 'G', 'G_random', 'delta_G', 'image_head_ci95', 'individual_reference_G']}
    gaussian_arrays = dict(arrays, peaks_xy=bank['peaks'], head_indices=bank['head_indices'])
    gaussian = evaluate(gaussian_arrays, config)
    reference_result['family_difference'] = {'mean': permuted['delta_G'] - gaussian['delta_G'],
                                           'image_head_ci95': np.quantile(permutation_joint - gaussian_joint, [.025, .975]).tolist()}
    species = load_arrays(data_root / 'extensions/donor_species.npz')['species']
    actual = {'channel_permutation': reference_result, 'donor_species': species_comparison(context, bank, species)}
    error = check_expected(actual, read_json(data_root / 'extensions/expected_results.json'))
    return {'status': 'passed', 'maximum_error': error, **actual}
