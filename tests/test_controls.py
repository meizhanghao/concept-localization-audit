"""Small control examples independent of the released numerical estimates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit.core import score_case
from audit.io import check_expected, verify_inputs
from audit.metrics import keypoint_hit


class CorrespondenceControls(unittest.TestCase):
    def make_case(self, root, trained):
        # Two separated targets: matched peaks always hit, swapped peaks miss.
        # A fixed peak has identical own/donor scores regardless of ownership.
        fixed = np.array([[20, 20], [20, 20]])
        np.savez(root / 'context.npz',
                 donors=np.array([[1, 0]]), image=np.array([0, 1]), group=np.array([0, 0]),
                 points_xy=np.array([[[20, 20]], [[200, 200]]]),
                 weights=np.array([[1., 1.], [2., 2.]]), head_indices=np.zeros((2, 1), dtype=int))
        np.savez(root / 'peaks.npz', peaks=np.array([trained, fixed]))
        return {'shared': 'context.npz', 'file': 'peaks.npz', 'joint': True}

    def test_image_paired_control_exceeds_fixed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_case(root, np.array([[20, 20], [200, 200]]))
            result, _ = score_case(case, root)
        self.assertEqual(result['S'], 1.)
        self.assertEqual(result['G'], 1.)
        self.assertEqual(result['G_random'], 0.)
        self.assertEqual(result['delta_G'], 1.)

    def test_image_independent_prior_has_zero_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.make_case(root, np.array([[20, 20], [20, 20]]))
            result, _ = score_case(case, root)
        self.assertEqual(result['S'], .5)
        self.assertEqual(result['G'], 0.)
        self.assertEqual(result['delta_G'], 0.)


class GeometryAndIntegrity(unittest.TestCase):
    def test_clipped_inclusive_window_and_invalid_keypoints(self):
        peak = np.array([[0, 0], [224, 224], [0, 0], [0, 0]])
        points = np.array([[[45, 45]], [[179, 179]], [[46, 45]], [[-1, -1]]])
        np.testing.assert_array_equal(keypoint_hit(peak, points), [True, True, False, False])

    def test_corrupted_and_missing_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / 'input.txt'
            payload.write_bytes(b'original input')
            checksum = hashlib.sha256(payload.read_bytes()).hexdigest()
            (root / 'checksums.json').write_text(json.dumps({'algorithm': 'sha256',
                                                            'files': {'input.txt': checksum}}))
            self.assertEqual(verify_inputs(root), 1)
            payload.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_inputs(root)
            payload.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing input'):
                verify_inputs(root)

    def test_nonfinite_or_different_results_fail(self):
        for value in (float('nan'), float('inf'), .1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_expected({'G': value}, {'G': 0.})


if __name__ == '__main__':
    unittest.main()
