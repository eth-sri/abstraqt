from operator import __xor__, __add__, __mul__, __sub__, __invert__, __or__, __and__, __eq__
from typing import Sequence, Callable

import numpy as np

from abstraqt.abstract_interpretation.interfaces.abstract_bit_pattern_array import AbstractBitPatternArray
from abstraqt.abstract_interpretation.interfaces.best_transformers import add_best_transformer_to_class
from abstraqt.linalg.solve_mod_2 import get_kernel_mod_2, solve_mod_2
from abstraqt.utils.array.mod_array import Mod2Array


class AbstractBoolArray(AbstractBitPatternArray):
    """
    Represents abstract booleans ⊆ {0,1} by:
    - 0 ≘ 00 ≘ {}
    - 1 ≘ 01 ≘ {0}
    - 2 ≘ 10 ≘ {1}
    - 3 ≘ 11 ≘ {0,1}
    """

    ################
    # CONSTRUCTORS #
    ################

    @staticmethod
    def zeros(shape: Sequence[int]):
        ret = np.zeros(shape, dtype=np.uint8)
        ret = AbstractBoolArray.lift(ret)
        return ret

    #############
    # CONSTANTS #
    #############

    zero_0D__representation = None
    one_0D__representation = None
    constants_to_load = {'zero': 0, 'one': 1}

    ########
    # JOIN #
    ########

    @staticmethod
    def empty_reference(shape: Sequence[int]):
        return Mod2Array.empty(shape)

    ##############
    # OPERATIONS #
    ##############

    bound_on_representation = 4

    def sum(self, axis=None):
        representation = sum_abstract_bool_array_representation(self.representation, axis=axis)
        return AbstractBoolArray(representation)

    def dot(self, other):
        assert isinstance(other, AbstractBoolArray)
        representation = dot__abstract_bool_array_representation(self.representation, other.representation)
        return AbstractBoolArray(representation)

    def all(self, axis=None):
        ret = all__abstract_bool_array_representation(self.representation, axis=axis)
        return AbstractBoolArray(ret)

    def to_abstract_pauli_bit_representation(self):
        # will be overwritten by AbstractPauliBitRepresentation
        raise NotImplementedError()


##############
# OPERATIONS #
##############


xor__abstract_bool_array_representation = add_best_transformer_to_class(AbstractBoolArray, '__xor__', __xor__)
add_best_transformer_to_class(AbstractBoolArray, '__add__', __add__)
mul_abstract_bool_representation = add_best_transformer_to_class(AbstractBoolArray, '__mul__', __mul__)
add_best_transformer_to_class(AbstractBoolArray, '__sub__', __sub__)
add_best_transformer_to_class(AbstractBoolArray, '__and__', __and__)
add_best_transformer_to_class(AbstractBoolArray, '__or__', __or__)
add_best_transformer_to_class(AbstractBoolArray, '__eq__', __eq__)
invert_abstract_bool_representation = add_best_transformer_to_class(AbstractBoolArray, '__invert__', __invert__)


# cannot apply numba because it cannot handle axis for "any"
def sum_abstract_bool_array_representation(representation: np.ndarray, axis=None):
    one = np.array(1, dtype=np.uint8)

    # fix concrete values correctly (except in case of bottom/top)
    ones = np.sum(representation == 2, axis=axis) % 2
    ret = np.asarray(ones + one, dtype=np.uint8)

    # fix tops
    tops = np.any(representation == 3, axis=axis)
    ret[tops] = 3

    # fix bottoms
    bottoms = np.any(representation == 0, axis=axis)
    ret[bottoms] = 0

    return ret


def dot_abstract_generic(x: np.ndarray, y: np.ndarray, mul_abstract: Callable, sum_abstract: Callable):
    assert len(y.shape) > 0

    # ensure other stores values to be multiplied on last dimension
    if len(y.shape) == 1:
        was_1d = True
        y = y.reshape((1, y.shape[0]))
    else:
        was_1d = False
        y = np.swapaxes(y, -1, -2)

    # ensure last dimension matches
    assert x.shape[-1] == y.shape[-1]

    n_x = len(x.shape) - 1
    n_y = len(y.shape) - 1

    x = x.reshape(x.shape[:-1] + n_y * (1,) + (x.shape[-1],))
    y = y.reshape(n_x * (1,) + y.shape)

    ret = mul_abstract(x, y)
    ret = sum_abstract(ret, axis=-1)

    if was_1d:
        ret = ret.reshape((ret.shape[0],))

    return ret


def dot__abstract_bool_array_representation(x: np.ndarray, y: np.ndarray):
    return dot_abstract_generic(x, y, mul_abstract_bool_representation, sum_abstract_bool_array_representation)


def all__abstract_bool_array_representation(x: np.ndarray, axis=None):
    ret = np.zeros_like(x, dtype=np.uint8)
    ret = np.any(ret, axis=axis)
    ret = np.asarray(ret, dtype=np.uint8)
    ret[...] = AbstractBoolArray.one_0D__representation
    # invariant: if ret[index] should be true, it is

    tops = np.any(x == AbstractBoolArray.top_0D__representation, axis=axis)
    ret[tops] = AbstractBoolArray.top_0D__representation
    # invariant: if ret[index] should be true/top, it is

    falses = np.any(x == AbstractBoolArray.zero_0D__representation, axis=axis)
    ret[falses] = AbstractBoolArray.zero_0D__representation
    # invariant: if ret[index] should be true/top/false, it is

    bottoms = np.any(x == AbstractBoolArray.bottom_0D__representation, axis=axis)
    ret[bottoms] = AbstractBoolArray.bottom_0D__representation
    # invariant: ret is correct

    return ret


#########
# SOLVE #
#########


def solution_space_mod_2_abstract__representation(a: np.ndarray, b: np.ndarray):
    """
    Return (x0, K), where all solutions x with a @ x = b can be generated by x0 + K @ s, for a boolean vector s

    Return (None, _) if a @ x = b is not satisfiable
    """
    # prepare inputs
    a = a.astype(np.uint8)
    if len(b.shape) == 1:
        one_d = True
        b = np.atleast_2d(b).T
    else:
        one_d = False

    # handle bottom
    if np.any(b == AbstractBoolArray.bottom_0D__representation):
        return None, None

    # rows with top represent useless constraints -> drop these rows
    concrete = b != AbstractBoolArray.top_0D__representation
    concrete = np.squeeze(concrete, axis=1)
    a = a[concrete, :]
    b = b[concrete]

    # concretize
    b = np.log2(b).astype(np.uint8)

    # solve
    x = solve_mod_2(a, b)

    if x is None:
        return None, None

    kernel = get_kernel_mod_2(a)

    if one_d:
        x = np.reshape(x, (-1,))
    else:
        x = np.reshape(x, (-1, 1))

    return x, kernel


def solution_space_mod_2_abstract(a: np.ndarray, b: AbstractBoolArray):
    ret = solution_space_mod_2_abstract__representation(a, b.representation)
    return ret


def solve_mod_2_abstract_rhs(a: np.ndarray, b: AbstractBoolArray):
    """
    Return a representation of all x with a @ x = b
    """
    x, kernel = solution_space_mod_2_abstract(a, b)

    if x is None:
        shape = (a.shape[1],)
        if len(b.shape) > 1:
            shape += (1,)
        ret = AbstractBoolArray.bottom(shape)
        return ret

    # abstract solution
    x_representation = 1 << x
    x_representation = x_representation.astype(np.uint8)

    # move ambiguous solutions to top
    tops = np.any(kernel, axis=1)
    x_representation[tops] = AbstractBoolArray.top_0D__representation

    ret = AbstractBoolArray(x_representation)

    return ret
