"""Check tied ranks, undefined correlations and score ownership independently."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/renderer_validity'))
from renderer_statistics import (weighted_spearman, macro_correlation,
                                 scores_from_peaks, selection_utility)


def explicit_spearman(x, y):
    # O(n^2) reference avoids the weighted implementation and its group logic.
    def rank(a):
        return np.array([(a < value).sum()+.5*(a == value).sum() for value in a])
    x, y = rank(x), rank(y)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


class RendererCriterionChecks(unittest.TestCase):
    def test_weighted_tied_ranks_equal_explicit_image_repetition(self):
        rng = np.random.default_rng(19)
        x, y = rng.integers(0, 3, 12), rng.integers(-2, 4, 12)
        weights = rng.multinomial(12, np.full(12, 1/12), size=40)
        expected = [explicit_spearman(np.repeat(x, row), np.repeat(y, row))
                    for row in weights]
        np.testing.assert_allclose(weighted_spearman(x, y, weights), expected,
                                   atol=1e-12, rtol=0, equal_nan=True)

    def test_constant_part_does_not_silently_leave_macro_statistic(self):
        x = np.tile(np.arange(12)[:, None], (1, 5)).astype(float)
        y = x.copy(); x[:, 0] = 1
        summary, draws, _ = macro_correlation(x, y, np.ones_like(x, bool),
                                              np.ones((3, 12)))
        self.assertIsNone(summary['estimate'])
        self.assertEqual(summary['valid_bootstrap_fraction'], 0)
        self.assertTrue(np.isnan(draws).all())
        self.assertEqual(summary['per_part'][1:], [1., 1., 1., 1.])

    def test_score_transfer_keeps_recipient_window(self):
        # Own peaks hit disjoint recipient targets. Swapping image owners misses.
        side, n = 3, 2
        windows = np.zeros((n, 5, side*side), bool)
        windows[0, :, 0] = True; windows[1, :, 8] = True
        peaks = np.zeros((3, 2, 4, n, 5, 2), dtype=int)
        peaks[:, :, 0, 1] = 2  # trained peaks follow images; random peaks fixed
        donors = np.broadcast_to(np.array([1, 0])[None, :, None], (1, n, 5))
        data = {'side': side, 'peaks': peaks,
                'windows_packed': np.packbits(windows, axis=-1)}
        scores = scores_from_peaks(data, np.arange(n), donors)
        np.testing.assert_array_equal(scores[:, :, 0], 1)
        np.testing.assert_array_equal(scores[:, :, 1], 1)
        np.testing.assert_array_equal(scores[:, :, 2], 1)

    def test_uniform_ties_have_zero_advantage_and_positive_regret(self):
        response = np.tile(np.arange(5), (4, 1)).astype(float)
        result = selection_utility(np.ones_like(response), response,
                                   np.ones_like(response, bool), np.ones((3, 4)))
        self.assertEqual(result['selected_minus_uniform']['estimate'], 0.)
        self.assertEqual(result['regret']['estimate'], 2.)
        self.assertEqual(result['mean_tie_count'], 5.)

    def test_unchanged_render_has_exactly_zero_output_response(self):
        root = Path(__file__).resolve().parents[1]/'experiments/renderer_validity'
        data = np.load(root/'data/inputs.npz', allow_pickle=False)
        response = data['logits'][:, :, None]-data['deleted_logits']
        self.assertTrue(data['unchanged'].any())
        self.assertEqual(float(abs(response[:, data['unchanged']]).max()), 0.)


if __name__ == '__main__':
    unittest.main()
