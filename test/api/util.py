# -*- coding: UTF-8 -*-

import numpy as np

from lava.api.util import mask_to_bounds, merge_bounds
from test.api.base import GlslBasedTest


class TestUtil(GlslBasedTest):

    def test_mask_to_bounds_single_byte(self):
        mask = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
        expected_bounds = [(2, 4)]
        np.testing.assert_array_equal(expected_bounds, mask_to_bounds(mask))

    def test_mask_to_bounds_bytes_at_beginning(self):
        mask = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        expected_bounds = [(0, 2)]
        np.testing.assert_array_equal(expected_bounds, mask_to_bounds(mask))

    def test_mask_to_bounds_bytes_at_end(self):
        mask = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
        expected_bounds = [(2, 6)]
        np.testing.assert_array_equal(expected_bounds, mask_to_bounds(mask))

    def test_mask_to_bounds_everything(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
        expected_bounds = [(0, 2), (3, 6), (7, 8), (9, 10), (13, 14)]
        np.testing.assert_array_equal(expected_bounds, mask_to_bounds(mask))

    def test_merge_bounds(self):
        input_bounds = [(0, 1), (4, 6), (7, 8), (9, 10), (13, 14)]
        expected_bounds = [(0, 1), (4, 10), (13, 14)]
        np.testing.assert_array_equal(expected_bounds, merge_bounds(input_bounds, 2))
