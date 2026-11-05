"""Coordinate regression tests; no weights, downloads or CUDA device required.

Run from the repository root with the normal OneMap dependencies installed:
    python -m unittest discover -s tests -p 'test_yolov7_coordinates.py' -v
"""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vision_models import yolov7_model


class YOLOv7CoordinateTests(unittest.TestCase):
    def detect_boxes(self, shape, boxes, image_size=640):
        detector = yolov7_model.YOLOv7Detector.__new__(yolov7_model.YOLOv7Detector)
        detector.image_size = image_size
        detector.half_precision = False
        detector.confidence_threshold = 0.8
        detector.classes = ['chair']
        chair = yolov7_model.COCO_CLASSES.index('chair')
        detector.classes_oi = [chair]
        detector.model = lambda image: (torch.empty((0, 6)),)
        predictions = torch.tensor([list(box)+[0.95, chair] for box in boxes],
                                   dtype=torch.float32).reshape(-1, 6)
        # Only the inference result and device transfer are replaced. Exercise
        # detect()'s real image preprocessing, box restoration and final filters.
        with patch.object(torch.Tensor, 'to', lambda tensor, *args, **kwargs: tensor), \
                patch.object(yolov7_model, 'non_max_suppression', return_value=[predictions]), \
                contextlib.redirect_stdout(io.StringIO()):
            return detector.detect(np.zeros((*shape, 3), dtype=np.uint8))['boxes']

    def test_square_camera_chair_at_right_of_image(self):
        # Actual pre-restoration detection from the HM3D chair failure.
        actual = self.detect_boxes((640, 640), [[499.5, 218.625, 554.5, 288.0]])
        self.assertEqual(actual, [[500.0, 312.0, 554.0, 411.0]])

    def test_square_camera_box_does_not_shift_towards_edge(self):
        actual = self.detect_boxes((640, 640), [[486.0, 219.25, 536.0, 278.25]])
        self.assertEqual(actual, [[486.0, 313.0, 536.0, 398.0]])

    def test_right_edge_box_does_not_collapse(self):
        actual = self.detect_boxes((640, 640), [[600.0, 70.0, 630.0, 140.0]])
        self.assertEqual(actual, [[600.0, 100.0, 630.0, 200.0]])

    def test_landscape_camera_scales_each_axis(self):
        actual = self.detect_boxes((480, 640), [[100.0, 112.0, 500.0, 336.0]])
        self.assertEqual(actual, [[100.0, 120.0, 500.0, 360.0]])

    def test_portrait_camera_scales_each_axis(self):
        actual = self.detect_boxes((640, 480), [[160.0, 112.0, 480.0, 336.0]])
        self.assertEqual(actual, [[120.0, 160.0, 360.0, 480.0]])

    def test_odd_camera_dimensions(self):
        actual = self.detect_boxes((333, 517), [[128.0, 112.0, 512.0, 336.0]])
        self.assertEqual(actual, [[103.0, 83.0, 414.0, 250.0]])

    def test_letterbox_padding_is_removed_before_stretch(self):
        # A 608x425 resized image receives 11.5 px vertical letterbox padding.
        actual = self.detect_boxes((480, 640), [[152.0, 117.75, 456.0, 330.25]],
                                   image_size=608)
        self.assertEqual(actual, [[160.0, 120.0, 480.0, 360.0]])

    def test_boxes_are_clipped_to_original_pixel_bounds(self):
        actual = self.detect_boxes((640, 640), [[-10.0, -10.0, 650.0, 458.0]])
        self.assertEqual(actual, [[0.0, 0.0, 639.0, 639.0]])

    def test_empty_detections(self):
        self.assertEqual(self.detect_boxes((640, 640), []), [])


if __name__ == '__main__':
    unittest.main()
