import galois
import numpy as np

from abstraqt.linalg.solve_mod_2.solve_helper import ensure_b_column, solve_via_gaussian_elimination, \
    kernel_via_gaussian_elimination

GF2 = galois.GF(2)


@ensure_b_column
def solve_mod_2_full_galois(a: np.ndarray, b: np.ndarray):
    assert a.dtype == np.uint8
    assert b.dtype == np.uint8

    a = GF2(a)
    b = GF2(b)

    x = np.linalg.solve(a, b)
    ret = np.array(x, dtype=np.uint8)

    return ret


def gaussian_elimination_galois(a: np.ndarray):
    a_gf2 = GF2(a)
    a_gf2 = a_gf2.row_reduce()
    a = np.array(a_gf2, dtype=np.uint8)
    return a


solve_mod_2_galois = solve_via_gaussian_elimination(gaussian_elimination_galois)
get_kernel_mod_2_galois = kernel_via_gaussian_elimination(gaussian_elimination_galois)
