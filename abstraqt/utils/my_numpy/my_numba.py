import numpy as np
from numba import njit

from abstraqt.utils.string_helper import get_environment_variable_bool

use_numba = get_environment_variable_bool('USE_NUMBA', default=True)

if use_numba:
    my_njit = njit(cache=True)
else:
    def identity(x):
        return x


    my_njit = identity


@my_njit
def outer_with_and(a: np.ndarray, b: np.ndarray):
    """
    Assumes a and b are flattened
    """

    a = np.atleast_2d(a)
    a = a.T
    # other versions:
    # a = np.reshape(a, newshape=(-1, 1))
    # a = a[:, np.newaxis]
    # a = np.expand_dims(a, axis=-1)
    return np.bitwise_and(a, b)
