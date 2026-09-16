"""Synthetic geometry checks; no detector accuracy or physical validity claims."""
import unittest

from secondlook.anomaly import (AnomalyContractError, ObservationInvalidError,
                               _foreground_box, _foreground_config, _prepare)

try:
    import cv2
    import numpy as np
    from PIL import Image
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires installed Intel image stack: OpenCV, NumPy, Pillow')
class ForegroundCropTests(unittest.TestCase):
    def setUp(self):
        self.config = _foreground_config()

    def test_effective_geometry_and_black_mark_pixels_are_preserved(self):
        image = np.full((200, 200, 3), 130, dtype=np.uint8)
        image[70:130, 70:130] = [230, 30, 20]
        image[90:105, 90:105] = 0
        cropped, geometry = _prepare(image, 64, [.25, .25, .75, .75], self.config)
        expected, _ = _prepare(image[65:135, 65:135], 64)
        np.testing.assert_array_equal(cropped, expected)
        self.assertEqual(geometry['parent_roi_pixels'], [50, 50, 150, 150])
        self.assertEqual(geometry['roi_pixels'], [65, 65, 135, 135])
        self.assertFalse(geometry['foreground_crop']['mask_applied_to_model_input'])
        self.assertEqual(geometry['foreground_crop']['component_area'], 3600 - 225)

    def test_absent_foreground_is_invalid(self):
        image = np.full((100, 100, 3), 130, dtype=np.uint8)
        with self.assertRaisesRegex(ObservationInvalidError, 'absent or too small'):
            _foreground_box(Image.fromarray(image), self.config)

    def test_comparable_components_are_invalid(self):
        image = np.full((100, 100, 3), 130, dtype=np.uint8)
        image[10:40, 10:40] = [220, 20, 20]
        image[60:80, 60:80] = [20, 220, 20]
        with self.assertRaisesRegex(ObservationInvalidError, 'multiple comparable'):
            _foreground_box(Image.fromarray(image), self.config)

    def test_tiny_component_or_thin_box_is_invalid(self):
        for shape in ((5, 5), (10, 80)):
            image = np.full((100, 100, 3), 130, dtype=np.uint8)
            image[10:10 + shape[0], 10:10 + shape[1]] = [220, 20, 20]
            with self.assertRaisesRegex(ObservationInvalidError, 'too small'):
                _foreground_box(Image.fromarray(image), self.config)

    def test_padding_clamps_to_parent(self):
        image = np.full((100, 100, 3), 130, dtype=np.uint8)
        image[0:60, 0:60] = [220, 20, 20]
        box, _ = _foreground_box(Image.fromarray(image), self.config)
        self.assertEqual(box, [0, 0, 65, 65])

    def test_disabled_foreground_preserves_default_preprocessing(self):
        image = np.full((100, 100, 3), 130, dtype=np.uint8)
        first, _ = _prepare(image, 64)
        second, _ = _prepare(image, 64, None, None)
        np.testing.assert_array_equal(first, second)

    def test_configuration_errors_are_not_invalid_observations(self):
        image = np.full((100, 100, 3), 130, dtype=np.uint8)
        with self.assertRaises(AnomalyContractError) as raised:
            _foreground_box(Image.fromarray(image), {'saturation_min': -1})
        self.assertNotIsInstance(raised.exception, ObservationInvalidError)


if __name__ == '__main__':
    unittest.main()
