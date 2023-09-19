from unittest import TestCase

import numpy as np

from abstraqt.utils.my_array.lookup_table import LookupTable


def reference_implementation_1(x):
    return ~x & 15


def reference_implementation_2(x, y):
    return ~x & y


class TestLookupTable(TestCase):

    def test_single_arg(self):
        t = LookupTable(reference_implementation_1, [16], 'test.one_arg')

        result = t(1)
        self.assertEqual(result, 14)

        results = t(np.array([1, 1]))
        np.testing.assert_equal(results, [14, 14])

    def test_two_args(self):
        t = LookupTable(reference_implementation_2, [16, 16], 'test.two_args')

        result = t(0, 1)
        self.assertEqual(result, 1)

        results = t(np.array([0, 0]), np.array([1, 1]))
        np.testing.assert_equal(results, [1, 1])
