from functools import reduce
from typing import List

import numpy as np


random_number_generator = np.random.default_rng(0)


np.seterr(all='raise')
# temporarily disable with:
# with np.errstate(over='ignore', under='ignore'):


def concatenate_last_axis(a: np.ndarray, axis=0):
    t = tuple(a[..., i] for i in range(a.shape[-1]))
    ret = np.concatenate(t, axis=axis)
    return ret


def bitwise_implies(a: np.ndarray, b: np.ndarray):
    return np.bitwise_or(
        np.bitwise_not(a),
        b
    )


def is_bitwise_one(a: np.ndarray):
    with np.errstate(over='ignore'):
        return a + np.ones_like(a) == 0


def matrix_product(matrices: List[np.ndarray]):
    return reduce(np.dot, matrices)


def matmul_mod2(a: np.ndarray, b: np.ndarray):
    t = a.dtype
    ret = a.astype(np.uint8) @ b.astype(np.uint8)
    ret %= 2
    return ret.astype(t)


def map_to_consecutive(a: np.ndarray):
    """
    Given a flat array a, returns:
    - a number n_groups indicating how many unique elements are contained in a
    - an array s such that:
      - s[i] = s[j] <=> a[i] = a[j]
      - s is a permutation of numbers between 0 and n_groups
    """
    u, indices = np.unique(a, return_inverse=True)
    n_groups, = u.shape
    return n_groups, indices


def invert_permutation(p):
    """
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    # https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy

    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def ravel_multi_index(multi_index, dims):
    """
    https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
    """
    assert len(dims) == multi_index.shape[0]
    if len(dims) == 0:
        return np.zeros(multi_index.shape[1], dtype=int)
    else:
        return np.ravel_multi_index(multi_index, dims)


count_one_bits_table = np.array([bin(i).count('1') for i in range(1 << 8)]).astype(int)


def count_one_bits(x: np.ndarray):
    assert x.dtype == np.uint8
    return count_one_bits_table[x]
