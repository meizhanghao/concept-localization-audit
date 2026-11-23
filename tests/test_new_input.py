"""Independent small examples for new-input scoring and protocol validation."""
import copy
import unittest

import numpy as np

from audit.new_input import evaluate


class NewInputAudit(unittest.TestCase):
    def example(self):
        arrays = {'peaks_xy': np.array([[[1, 1], [8, 8]], [[1, 1], [1, 1]]]),
                  'image_id': np.array([100, 200]), 'group_id': np.array(['part', 'part']),
                  'concept_id': np.array(['red', 'red']), 'donors': np.array([[1, 0]]),
                  'points_xy': np.array([[[1, 1]], [[8, 8]]]),
                  'bootstrap_weights': np.array([[2, 0], [1, 1], [0, 2]])}
        config = {'geometry': {'type': 'keypoints', 'side': 10, 'half_width': 0}, 'seed': 1}
        return arrays, config

    def test_new_data_without_expected_results(self):
        arrays, config = self.example(); result = evaluate(arrays, config)
        self.assertEqual((result['S'], result['G'], result['G_random'], result['delta_G']), (1., 1., 0., 1.))
        self.assertEqual(result['image_head_ci95'], [1., 1.])
        self.assertEqual(result['cells'], {'trained_own': 1., 'trained_donor': 0., 'reference_own': .5, 'reference_donor': .5})
        self.assertEqual(result['spatial_requirement'], 'not specified')
        self.assertEqual(result['chance'], .01)

    def test_mask_and_keypoint_geometries_agree(self):
        arrays, config = self.example(); expected = evaluate(arrays, config)
        arrays['windows'] = np.zeros((2, 10, 10), dtype=bool)
        arrays['windows'][0, 1, 1] = True; arrays['windows'][1, 8, 8] = True
        config['geometry'] = {'type': 'mask', 'side': 10}
        result = evaluate(arrays, config)
        for key in ('S', 'G', 'G_random', 'delta_G', 'chance'):
            self.assertEqual(result[key], expected[key])

    def test_same_image_or_wrong_concept_donors_are_rejected(self):
        arrays, config = self.example(); arrays['donors'][0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'different image'):
            evaluate(arrays, config)
        arrays, config = self.example(); arrays['concept_id'][1] = 'blue'
        with self.assertRaisesRegex(ValueError, 'concept identity'):
            evaluate(arrays, config)

    def test_noninteger_or_out_of_frame_peaks_are_rejected(self):
        arrays, config = self.example(); arrays['peaks_xy'] = arrays['peaks_xy'].astype(float)
        with self.assertRaisesRegex(ValueError, 'integers'):
            evaluate(arrays, config)
        arrays, config = self.example(); arrays['peaks_xy'][0, 0] = [10, 1]
        with self.assertRaisesRegex(ValueError, 'inside'):
            evaluate(arrays, config)

    def test_multiple_concepts_within_one_macro_group_are_counted(self):
        arrays, config = self.example()
        arrays['peaks_xy'] = np.repeat(arrays['peaks_xy'], 2, axis=1)
        arrays['image_id'] = np.array([100, 100, 200, 200])
        arrays['group_id'] = np.array(['part'] * 4)
        arrays['concept_id'] = np.array(['red', 'long', 'red', 'long'])
        arrays['points_xy'] = np.repeat(arrays['points_xy'], 2, axis=0)
        arrays['donors'] = np.array([[2, 3, 0, 1]])
        result = evaluate(arrays, config)
        self.assertEqual(result['delta_G'], 1.)
        self.assertEqual(result['coverage']['group_counts'], {'part': 4})

    def test_exact_clipped_chance_matches_enumerated_grid(self):
        arrays, config = self.example()
        arrays['points_xy'] = np.array([[[0, 0], [1, 1]], [[10, 10], [-1, -1]]])
        config['geometry']['half_width'] = 2
        result = evaluate(arrays, config)
        yy, xx = np.mgrid[:10, :10]
        areas = []
        for points in arrays['points_xy']:
            low = np.maximum(np.stack([xx, yy], -1)[..., None, :] - 2, 0)
            high = np.minimum(np.stack([xx, yy], -1)[..., None, :] + 2, 10)
            areas.append(((points >= low) & (points <= high)).all(-1).any(-1).mean())
        self.assertAlmostEqual(result['chance'], np.mean(areas))

    def test_missing_group_draw_is_not_silently_dropped(self):
        arrays, config = self.example(); arrays['group_id'] = np.array(['one', 'two'])
        with self.assertRaisesRegex(ValueError, 'loses a declared macro group'):
            evaluate(arrays, config)


if __name__ == '__main__':
    unittest.main()
