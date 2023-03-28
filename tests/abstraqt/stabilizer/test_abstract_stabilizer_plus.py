from unittest import TestCase

import numpy as np

from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.iterator_helper import get_unique
from tests.test_config import default_random_repetitions


class TestAbstractStabilizerPlus(TestCase):

    def test_get_density_matrix(self):
        for n_bits in [1, 2, 3]:
            for seed in range(default_random_repetitions):
                with self.subTest(n_bits=n_bits, seed=seed):
                    g = AbstractStabilizerPlus.zero_state(n_bits)
                    m = get_unique(g.get_densities())
                    expected = np.zeros((1 << n_bits, 1 << n_bits))
                    expected[0, 0] = 1
                    np.testing.assert_array_equal(m, expected)
