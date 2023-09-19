from typing import Callable

import numpy as np

from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.my_numpy.my_numba import my_njit


def ensure_b_column(f: Callable[[np.ndarray, np.ndarray], np.ndarray]):

    @rename_function(f.__name__)
    def f_wrapped(a: np.ndarray, b: np.ndarray):
        if len(b.shape) == 1:
            one_d = True
            b = np.atleast_2d(b).T
        else:
            one_d = False

        x = f(a, b)

        if x is None:
            return x

        if one_d:
            x = np.reshape(x, (-1,))
        else:
            x = np.reshape(x, (-1, 1))

        return x

    return f_wrapped


def solve_via_gaussian_elimination(gaussian_elimination: Callable[[np.ndarray], np.ndarray]):

    @rename_function('solve_via_' + gaussian_elimination.__name__)
    @ensure_b_column
    def solve(a: np.ndarray, b: np.ndarray):
        ab = np.hstack((a, b))
        ix = gaussian_elimination(ab)

        # find unsatisfiable constraints
        zero_rows = np.all(ix[:, :-1] == 0, axis=1)
        if np.any(ix[zero_rows, -1]):
            return None

        x = _compose_solution(ix, zero_rows)
        return x

    return solve


@my_njit
def _compose_solution(ix: np.ndarray, zero_rows: np.ndarray):
    # drop zero rows
    ix = ix[~zero_rows, :]
    n, m = ix.shape
    m -= 1  # account for last column b

    # compose solution
    x = np.zeros((m, 1), dtype=np.uint8)
    for i in range(n):
        row = n - i - 1
        column = np.argmax(ix[row, :-1])

        x[column] = ix[row, -1]

    return x


def kernel_via_gaussian_elimination(gaussian_elimination: Callable[[np.ndarray], np.ndarray]):
    # Implementation inspired by
    # https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination

    @rename_function('kernel_via_' + gaussian_elimination.__name__)
    def kernel(a: np.ndarray):
        n_rows, n_columns = a.shape
        i = np.eye(n_columns, dtype=np.uint8)

        ai = np.vstack((a, i))

        # get column echelon form
        bc = gaussian_elimination(ai.T)
        bc = bc.T

        zero_column_b = ~np.any(bc[:n_rows, :], axis=0)
        non_zero_columns_c = np.any(bc[n_rows:, :], axis=0)
        columns = zero_column_b & non_zero_columns_c

        ret = bc[n_rows:, columns]

        return ret

    return kernel
