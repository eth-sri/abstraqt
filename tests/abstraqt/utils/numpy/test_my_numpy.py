from unittest import TestCase

import numpy as np

from abstraqt.utils.numpy.my_numpy import bitwise_implies, invert_permutation, map_to_consecutive, \
    count_one_bits


class TestMyNumpy(TestCase):

    def test_bitwise_implies(self):
        a = np.array(3, dtype=np.uint8)
        b = np.array(2, dtype=np.uint8)
        actual = bitwise_implies(a, b)

        for i in range(8):
            a_bit = (a >> i) & 1
            b_bit = (b >> i) & 1
            expected_bit = (not a_bit) or b_bit

            actual_bit = (actual >> i) & 1

            self.assertEqual(expected_bit, actual_bit)

    def test_invert_permutation(self):
        p = np.array([2, 0, 1])
        s = invert_permutation(p)

        expected = np.array([1, 2, 0])
        np.testing.assert_equal(s, expected)

    def test_map_to_consecutive(self):
        a = np.array([0, 1, 3, 1, 0])
        n_groups, s = map_to_consecutive(a)

        self.assertEqual(n_groups, 3)

        expected = np.array([0, 1, 2, 1, 0])
        np.testing.assert_equal(s, expected)

    def test_count_one_bits(self):
        a = np.array([1, 3, 0, 16], dtype=np.uint8)
        counts = count_one_bits(a)
        np.testing.assert_equal(counts, [1, 2, 0, 1])
