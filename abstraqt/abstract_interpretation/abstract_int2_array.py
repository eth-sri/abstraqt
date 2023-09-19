from operator import __add__, __mul__, __sub__, __neg__, __eq__
from typing import Sequence

import numpy as np

from abstraqt.abstract_interpretation.interfaces.abstract_bit_pattern_array import AbstractBitPatternArray
from abstraqt.abstract_interpretation.interfaces.best_transformers import add_best_transformer_to_class
from abstraqt.utils.my_array.lookup_table import LookupTable
from abstraqt.utils.my_array.mod_array import Mod4Array
from .abstract_bool_array import dot_abstract_generic, AbstractBoolArray


class AbstractInt2Array(AbstractBitPatternArray):
    """
    Represents abstract booleans ⊆ {0,1,2,3} by:
    - 0000 ≘ {}
    - 0001 ≘ {0}
    - 0010 ≘ {1}
    - 0100 ≘ {2}
    - 1000 ≘ {3}
    - ...
    - 15 ≘ 1111 ≘ {0,1,2,3}
    """

    #############
    # CONSTANTS #
    #############

    zero_0D__representation = None
    one_0D__representation = None
    two_0D__representation = None
    three_0D__representation = None
    constants_to_load = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}

    ########
    # JOIN #
    ########

    @staticmethod
    def empty_reference(shape: Sequence[int]):
        return Mod4Array.empty(shape)

    ##############
    # OPERATIONS #
    ##############

    bound_on_representation = 16

    def sum(self, axis=None):
        ret = sum__abstract_int2_array_representation(self.representation, axis=axis)
        return AbstractInt2Array(ret)

    def sum_n_times(self, n):
        ret = sum_n_times(self.representation, np.minimum(n, 4), n % 4)
        ret = np.asarray(ret, dtype=np.uint8)
        return AbstractInt2Array(ret)

    def dot(self, other):
        assert isinstance(other, AbstractInt2Array)
        representation = dot__abstract_int2_array_representation(self.representation, other.representation)
        return AbstractInt2Array(representation)

    def exponent_base_i(self):
        # implementation should be overridden by AbstractComplexArray
        raise NotImplementedError()


add__abstract_int2_array_representation = add_best_transformer_to_class(AbstractInt2Array, '__add__', __add__)
mul__abstract_int2_array_representation = add_best_transformer_to_class(AbstractInt2Array, '__mul__', __mul__)
add_best_transformer_to_class(AbstractInt2Array, '__sub__', __sub__)
add_best_transformer_to_class(AbstractInt2Array, '__neg__', __neg__)
add_best_transformer_to_class(AbstractInt2Array, '__eq__', __eq__, target_class=AbstractBoolArray)


def sum_n_times_reference(representation: np.ndarray, n_min_3: np.ndarray, n_mod_4: np.ndarray):
    """
    returns representation + ... + representation, where the sum ranges over n elements
    where n satisfies:
    - min(n, 3) = n_min_3
    - n % 4 = n_mod_4

    Note that this is not the same as n*representation, for example 4*{0,1,2,3} = {0}
    """
    assert representation.shape == ()
    assert n_min_3.shape == ()
    assert n_mod_4.shape == ()

    previous_ret = None
    for n in range(100):
        if min(n, 3) == n_min_3 and n % 4 == n_mod_4:
            ret = np.array(1, np.uint8)  # encode 0
            for _ in range(n):
                ret = add__abstract_int2_array_representation(ret, representation)
            if previous_ret is not None:
                # sanity check to ensure the answer is consistent
                assert previous_ret == ret
            previous_ret = ret
    if previous_ret is None:
        # impossible constraints on n, return bottom
        return np.array(0, np.uint8)
    else:
        return previous_ret


sum_n_times = LookupTable(sum_n_times_reference, [16, 4, 4], 'abstract_int2_sum_n_times').lookup


# Warning: using numba here yields incorrect results
def sum__abstract_int2_array_representation(representation: np.ndarray, axis=None):
    ret = None
    for pattern in range(16):
        count = np.sum(representation == pattern, axis=axis)
        to_add = sum_n_times(pattern, np.minimum(count, 3), count % 4)
        if ret is None:
            ret = np.asarray(to_add, dtype=np.uint8)
        else:
            ret = np.asarray(add__abstract_int2_array_representation(ret, to_add), dtype=np.uint8)
    return ret


def dot__abstract_int2_array_representation(x: np.ndarray, y: np.ndarray):
    return dot_abstract_generic(x, y, mul__abstract_int2_array_representation, sum__abstract_int2_array_representation)


def to_int2__abstract_bool_array_representation(x: np.ndarray, times_two=False):
    if times_two:
        ret = mul__abstract_int2_array_representation(AbstractInt2Array.two_0D__representation, x)
        return ret
    else:
        return x
