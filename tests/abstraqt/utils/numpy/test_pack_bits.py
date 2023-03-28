from unittest import TestCase

import numpy as np

from abstraqt.utils.numpy.pack_bits import unpack_bits, pack_bits

packed = np.array([3, 2], dtype=np.uint8)
unpacked = np.array([[1, 1], [1, 0]], dtype=np.uint8)


class TestPackBits(TestCase):

    def test_unpack_bits(self):
        unpacked_actual = unpack_bits(packed, 2)
        np.testing.assert_equal(unpacked_actual, unpacked)

    def test_pack_bits(self):
        packed_actual = pack_bits(unpacked)
        np.testing.assert_equal(packed_actual, packed)
