from unittest import TestCase

import numpy as np

from abstraqt.linalg.solve_mod_2.gaussian_elimination_mod_2 import invert_matrix_mod_2
from abstraqt.utils.my_numpy.my_numpy import matmul_mod2
from tests.abstraqt.linalg.random_matrix import get_random_bool_matrices

sizes = [(2, 2), (3, 3), (4, 4), (3, 2), (3, 1), (4, 2)]
n_tests = 50


class GaussianEliminationTest(TestCase):

    def test_invert(self):
        for a in get_random_bool_matrices(sizes, n_tests, full_rank=True):
            with self.subTest(shape=a.shape, a=a):
                a_inv = invert_matrix_mod_2(a)
                self.assertEqual(a_inv.dtype, a.dtype)
                hopefully_eye = matmul_mod2(a_inv, a)
                expected_eye = np.eye(*a.shape, dtype=a.dtype)
                np.testing.assert_equal(hopefully_eye, expected_eye, err_msg=f'{a} @ {a_inv} = {hopefully_eye} != I')
