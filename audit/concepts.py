"""Same-part label contrasts under declared concept-presence eligibility."""
import numpy as np

from .io import check_expected, load_arrays, read_json

# The assignment bootstrap is frozen to the original experiment's seed.
ASSIGNMENT_RESAMPLING_SEED = 2026112222


def run(data_root):
    root = data_root / 'concept_eligibility'
    data = load_arrays(root / 'data.npz')
    attribute, images, weights = data['attribute'], data['image'], data['weights']
    image_count, group_count = data['observed'].shape
    denominator = weights @ data['observed']
    heads = data['heads']
    draw_indices = np.arange(len(weights))[:, None]
    assignment_count = data['wrong_own'].shape[1]
    assignments = np.random.default_rng(ASSIGNMENT_RESAMPLING_SEED).integers(
        0, assignment_count, (len(weights), assignment_count))

    def point(values):
        return float(np.mean([values[attribute == group].mean() for group in range(group_count)]))

    def bootstrap(values):
        matrix = np.zeros((image_count, group_count))
        matrix[images, attribute] = values
        return ((weights @ matrix) / denominator).mean(1)

    correct = data['correct_own'] - data['correct_donor']
    mismatched = data['wrong_own'] - data['wrong_donor']
    contrast = correct - mismatched.mean(1)
    draws = []
    for head in range(len(correct)):
        bank = np.stack([bootstrap(values) for values in mismatched[head]], axis=1)
        draws.append(bootstrap(correct[head]) - bank[draw_indices, assignments].mean(1))
    draws = np.array(draws).T
    adjusted_draws = draws[:, 0] - draws[:, 1:][draw_indices, heads].mean(1)
    primary = {'G_specific': point(contrast[0]),
               'ci95': np.quantile(draws[:, 0], [.025, .975]).tolist(),
               'adjusted': point(contrast[0] - contrast[1:].mean(0)),
               'adjusted_ci95': np.quantile(adjusted_draws, [.025, .975]).tolist()}

    pair_ids, pair_images, pair_weights = data['pair_id'], data['pair_image'], data['pair_weights']
    pair_count = len(pair_weights)
    observed = np.zeros((image_count, pair_count))
    np.add.at(observed, (pair_images, pair_ids), 1)
    pair_denominator = weights @ observed
    retained = (pair_denominator > 0).all(1)
    if not retained.any():
        raise ValueError('No bootstrap draw retains every eligible pair')

    def pair_point(values):
        return float(sum(pair_weights[pair] * values[pair_ids == pair].mean()
                         for pair in range(pair_count)))

    def pair_bootstrap(values):
        matrix = np.zeros_like(observed)
        np.add.at(matrix, (pair_images, pair_ids), values)
        return (((weights[retained] @ matrix) / pair_denominator[retained]) * pair_weights).sum(1)

    paired_gains = data['pair_own'] - data['pair_donor']
    paired_contrast = paired_gains[:, 0] - paired_gains[:, 1]
    pair_draws = np.array([pair_bootstrap(values) for values in paired_contrast]).T
    indices = np.arange(retained.sum())[:, None]
    adjusted = pair_draws[:, 0] - pair_draws[:, 1:][indices, heads[retained]].mean(1)
    both_present = {'G_specific': pair_point(paired_contrast[0]),
                    'ci95': np.quantile(pair_draws[:, 0], [.025, .975]).tolist(),
                    'adjusted': pair_point(paired_contrast[0] - paired_contrast[1:].mean(0)),
                    'adjusted_ci95': np.quantile(adjusted, [.025, .975]).tolist()}
    result = {'primary': primary, 'both_present': both_present}
    error = check_expected(result, read_json(root / 'expected.json'))
    result.update(status='passed', maximum_error=error,
                  retained_pair_draw_fraction=float(retained.mean()))
    return result
