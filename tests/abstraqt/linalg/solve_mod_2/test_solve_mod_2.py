import unittest
from unittest import TestCase

import numpy as np

from abstraqt.linalg.solve_mod_2 import \
    solve_mod_2_full_galois, \
    solve_mod_2_galois, \
    get_kernel_mod_2_galois, \
    solve_mod_2_gaussian_elimination, \
    get_kernel_mod_2_gaussian_elimination
from abstraqt.linalg.solve_mod_2.gaussian_elimination_mod_2 import gaussian_elimination_mod_2_packing, \
    solve_mod_2_gaussian_elimination_packing, get_kernel_mod_2_gaussian_elimination_packing

from tests.abstraqt.linalg.random_matrix import get_bool_matrix

n_tests = 50


class AbstractSolveMod2Wrapper:
    class AbstractSolveMod2(TestCase):

        def __init__(
                self,
                solve_mod_2_function,
                kernel_mod_2_function,
                *args,
                include_non_square=True,
                full_rank=True,
                **kwargs
        ):
            super().__init__(*args, **kwargs)

            self.solve_mod_2_function = solve_mod_2_function
            self.kernel_mod_2_function = kernel_mod_2_function

            self.sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]
            if include_non_square:
                self.sizes += [(3, 1), (4, 2), (2, 4)]
            self.full_rank = full_rank

        def test_solve_various(self):
            for n, m in self.sizes:
                for seed in range(n_tests):
                    np.random.seed(seed)
                    a = get_bool_matrix(n, m, full_rank=self.full_rank)
                    x = np.random.randint(0, 2, size=(a.shape[1],)).astype(np.int32)
                    self.check_solve(a, x, seed)

        def check_solve(self, a: np.ndarray, x: np.ndarray, seed: int):
            a = a.astype(np.uint8)
            x = x.astype(np.uint8)
            b = (a @ x) % 2
            with self.subTest(shape=a.shape, seed=seed, a=a, x=x, b=b):
                x_computed = self.solve_mod_2_function(a, b)
                self.assertIsNotNone(x_computed)
                self.assertEqual(x_computed.shape, x.shape)
                reconstructed_b = a.astype(np.uint8) @ x_computed.astype(np.uint8)
                reconstructed_b %= 2
                np.testing.assert_equal(reconstructed_b, b, err_msg=f"x={x}")

        def test_get_kernel(self):
            if self.kernel_mod_2_function is not None:
                for shape in self.sizes:
                    s = min(shape[0], shape[1])
                    for i in range(1, s):
                        for seed in range(n_tests):
                            np.random.seed(seed)
                            a = get_bool_matrix(*shape, rank=i)

                            self.check_get_kernel(a, i, seed)

        def check_get_kernel(self, a: np.ndarray, rank: int, seed: int):
            with self.subTest(shape=a.shape, a=a, rank=rank, seed=seed):
                a = a.astype(np.uint8)
                k = self.kernel_mod_2_function(a)
                self.assertEqual(k.shape[1], a.shape[1] - rank)
                for c in range(k.shape[1]):
                    v = k[:, c]
                    self.assertTrue(np.any(v != 0))
                    b = (a @ v) % 2
                    np.testing.assert_equal(b, 0)


class SolveMod2FullGalois(AbstractSolveMod2Wrapper.AbstractSolveMod2):

    def __init__(self, *args, **kwargs):
        super().__init__(solve_mod_2_full_galois, None, *args, include_non_square=False, **kwargs)


class SolveMod2Galois(AbstractSolveMod2Wrapper.AbstractSolveMod2):

    def __init__(self, *args, **kwargs):
        super().__init__(solve_mod_2_galois, get_kernel_mod_2_galois, *args, include_non_square=True, full_rank=False, **kwargs)


class SolveMod2GaussianElimination(AbstractSolveMod2Wrapper.AbstractSolveMod2):

    def __init__(self, *args, **kwargs):
        super().__init__(
            solve_mod_2_gaussian_elimination,
            get_kernel_mod_2_gaussian_elimination,
            *args,
            include_non_square=True,
            full_rank=False,
            **kwargs
        )


class SolveMod2GaussianEliminationPacked(AbstractSolveMod2Wrapper.AbstractSolveMod2):

    def __init__(self, *args, **kwargs):
        super().__init__(
            solve_mod_2_gaussian_elimination_packing,
            get_kernel_mod_2_gaussian_elimination_packing,
            *args,
            include_non_square=True,
            full_rank=False,
            **kwargs
        )
