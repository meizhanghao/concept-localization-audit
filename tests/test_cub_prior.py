"""Meaningful invariants for the training-only CUB position control."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'experiments/cub_prior/cub_prior_statistics.py'
SPEC = importlib.util.spec_from_file_location('cub_prior_statistics_test', SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PositionPriorChecks(unittest.TestCase):
    def test_training_mean_excludes_invisible_landmarks(self):
        points = np.array([[[.1, .2], [99., 99.]], [[.3, .4], [.5, .6]]])
        visible = np.array([[True, False], [True, True]])
        center, count = module.training_centers(points, visible, [[0, 1]])
        np.testing.assert_allclose(center, [[.3, .4]])
        np.testing.assert_array_equal(count, [3])

    def test_same_concept_prior_has_zero_own_donor_gain(self):
        peaks = np.array([[100, 90], [100, 90]])
        target_points = np.array([[[100, 90]], [[190, 180]]])
        own = module.hits(peaks, target_points)
        other = module.hits(peaks[[1, 0]], target_points)
        np.testing.assert_array_equal(own, [True, False])
        np.testing.assert_array_equal(own, other)

    def test_missing_attribute_draw_is_not_silently_dropped(self):
        values = np.array([1., 0.])
        weights = np.array([[1., 1.], [2., 0.]])
        result, valid = module.bootstrap(values, np.array([0, 1]),
                                          np.array([0, 1]), weights, groups=2)
        np.testing.assert_array_equal(valid, [True, False])
        self.assertEqual(result[0], .5)
        self.assertTrue(np.isnan(result[1]))


if __name__ == '__main__':
    unittest.main()
