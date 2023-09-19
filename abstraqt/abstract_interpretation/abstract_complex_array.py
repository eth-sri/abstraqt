from typing import Sequence

import numpy as np

from abstraqt.abstract_interpretation.interfaces.best_transformers import add_best_transformer_to_class
from abstraqt.utils.my_array.dimensions.representation_dimensions_wrapper import RepresentationDimensionsWrapper
from abstraqt.utils.my_numpy.lift_to_numpy_array import lift_to_numpy_array
from .abstract_int2_array import AbstractInt2Array
from .interval_array import _project_to_unit, _if_nan_then_zero, IntervalArray, \
    cos__interval_array_representation, mul__interval_array_representation, ensure_cyclic_interval_start_in_range, \
    is_super_set_of_element_wise__interval_array_representation, neg__interval_array_representation

atol = 1e-8


class AbstractComplexArray(IntervalArray):
    """
    Second-to-last dimension stores [r, φ], representing complex number exp(r + i*φ) = exp(r)*cos(φ) + i*exp(r)*sin(φ)

    Last dimension stores intervals [l, u] representing interval [l, u]
    """

    ################
    # CONSTRUCTORS #
    ################

    def __init__(self, representation: np.ndarray, check_valid_entries=True):
        super().__init__(representation, check_valid_entries=check_valid_entries)

        assert len(representation.shape) >= 2
        assert representation.shape[-2:] == (2, 2)
        RepresentationDimensionsWrapper.__init__(self, 2)

    @staticmethod
    def top(shape: Sequence[int]):
        ret = AbstractComplexArray.empty(shape)
        ret.representation[..., 0, 0] = -np.inf
        ret.representation[..., 0, 1] = np.inf
        ret.representation[..., 1, 0] = 0
        ret.representation[..., 1, 1] = 2 * np.pi
        return ret

    @staticmethod
    def zeros(shape: Sequence[int]):
        ones = np.zeros(shape, dtype=complex)
        ret = AbstractComplexArray.lift(ones)
        return ret

    @staticmethod
    def ones(shape: Sequence[int]):
        ones = np.ones(shape, dtype=complex)
        ret = AbstractComplexArray.lift(ones)
        return ret

    @staticmethod
    def empty(shape: Sequence[int]):
        shape = tuple(shape) + (2, 2)
        representation = np.empty(shape, dtype=float)
        return AbstractComplexArray(representation, check_valid_entries=False)

    @staticmethod
    def lift(x):
        if isinstance(x, AbstractComplexArray):
            return x

        x = lift_to_numpy_array(x)
        assert isinstance(x, np.ndarray)

        r, phi = AbstractComplexArray.get_r_phi(x)
        ret = AbstractComplexArray.empty(r.shape)
        ret.representation[..., 0, :] = np.expand_dims(r, -1)
        ret.representation[..., 1, :] = np.expand_dims(phi, -1)

        return ret

    @staticmethod
    def lift_log_of_real(x: np.ndarray):
        ret = AbstractComplexArray.empty(x.shape)
        ret.representation[..., 0, :] = np.expand_dims(x, -1)
        ret.representation[..., 1, :] = 0
        return ret

    @staticmethod
    def get_r_phi(x: np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.log(np.abs(x, dtype=float))
        phi = np.angle(x)
        phi = np.asarray(phi)
        # ensure phi lies in [0,2π]
        phi[phi < 0] += 2 * np.pi
        return r, phi

    @staticmethod
    def exponent_base_i(exponent: AbstractInt2Array):
        # overwritten later on
        pass

    ########
    # JOIN #
    ########

    # def join(self, other):
    #     # this "canonical" transformer would be sound but leads to failing tests (our tests exploit properties which are invalidated by this optimization)
    #     self = self.to_canonical()
    #     other = other.to_canonical()
    #     return IntervalArray.join(self, other)

    #############
    # CANONICAL #
    #############

    def to_canonical(self):
        self.representation = to_canonical__abstract_complex_array_representation(self.representation)
        return self

    ##############
    # OPERATIONS #
    ##############

    def __mul__(self, other):
        return IntervalArray.__add__(self, other)

    def __add__(self, other):
        raise NotImplementedError()

    def __neg__(self):
        raise NotImplementedError()

    def real(self):
        ret = real__abstract_complex_array_representation(self.representation)
        return IntervalArray(ret)

    def conjugate(self):
        ret = conjugate__abstract_complex_array_representation(self.representation.copy())
        ret = AbstractComplexArray(ret)
        return ret

    def sum(self, axis=None):
        raise NotImplementedError()

    ###########
    # HELPERS #
    ###########

    def equal_abstract_object_element_wise(self, other):
        other = self.lift(other)
        ret = np.isclose(self.representation, other.representation)
        ret = ret.all(axis=(-1, -2))
        return ret

    def is_super_set_of(self, other):
        assert isinstance(other, AbstractComplexArray)
        return is_super_set_of__abstract_complex_array_representation(self.representation, other.representation)

    def get_corners_single(self, numpy_array=True):
        assert self.representation.shape == (2, 2)
        if not self.is_bottom():
            rs = np.unique(self.representation[0, :])
            for r in rs:

                # add bounds
                phis = list(np.unique(self.representation[1, :]))

                # add "interesting" internal values: multiples of π
                next_lower_multiple = (self.representation[1, 0] // (2 * np.pi)) * (2 * np.pi)
                for i in [1, 2, 3]:
                    candidate = next_lower_multiple + i * np.pi
                    if self.representation[1, 0] < candidate < self.representation[1, 1]:
                        phis += [candidate]

                for phi in phis:
                    if numpy_array:
                        ret = np.exp(r + 1j * phi)
                    else:
                        ret = np.array([[r, r], [phi, phi]], dtype=float)
                        ret = AbstractComplexArray(ret)
                    yield ret

    @staticmethod
    def empty_reference(shape: Sequence[int], numpy_array=True):
        if numpy_array:
            return np.empty(shape, dtype=complex)
        else:
            return AbstractComplexArray.empty(shape)

    @staticmethod
    def repr_single(v):
        return f'r=[{v[0, 0]}, {v[0, 1]}], φ=[{v[1, 0]}, {v[1, 1]}]'


def to_canonical__abstract_complex_array_representation(representation: np.ndarray):
    """
    Ensure that:
    - If one element is bottom, all elements are bottom
    - If φ covers the full range, its interval is [0, 2π]
    - The lower bound for φ lies in [0, 2π]
    """

    # handle bottom
    if np.any(np.isnan(representation)):
        representation[...] = np.nan

    # handle top φ
    length_phi = representation[..., 1, 1] - representation[..., 1, 0]
    full_phi = length_phi >= 2 * np.pi - atol
    representation[full_phi, 1, 0] = 0  # triggers an error in numba if full_phi has multiple dimensions
    representation[full_phi, 1, 1] = 2 * np.pi

    # ensure φ interval starts in range [0, 2π]
    representation = ensure_phi_in_range(representation)

    return representation


def is_super_set_of__abstract_complex_array_representation(super_representation: np.ndarray,
                                                           sub_representation: np.ndarray):
    if np.any(np.isnan(sub_representation)):
        return True
    else:
        super_representation = to_canonical__abstract_complex_array_representation(super_representation)
        sub_representation = to_canonical__abstract_complex_array_representation(sub_representation)

        super1 = is_super_set_of_element_wise__interval_array_representation(super_representation, sub_representation)
        # sometimes subsets can only be detected by shifting
        sub_representation[..., 1, :] += 2 * np.pi
        super2 = is_super_set_of_element_wise__interval_array_representation(super_representation, sub_representation)

        return np.all(np.logical_or(super1, super2))


##############
# OPERATIONS #
##############


def real__abstract_complex_array_representation(representation: np.ndarray):
    cos = cos__interval_array_representation(representation[..., 1, :])
    length = np.exp(representation[..., 0, :])
    result = mul__interval_array_representation(length, cos)
    return result


def conjugate__abstract_complex_array_representation(representation: np.ndarray):
    representation[..., 1, :] = neg__interval_array_representation(representation[..., 1, :])
    return representation


def ensure_phi_in_range(representation: np.ndarray):
    representation[..., 1, :] = ensure_cyclic_interval_start_in_range(representation[..., 1, :])
    return representation


def power_of_i(x: AbstractInt2Array):
    return np.power(1j, x.representation)


add_best_transformer_to_class(
    AbstractInt2Array,
    'exponent_base_i',
    power_of_i,
    target_class=AbstractComplexArray
)
