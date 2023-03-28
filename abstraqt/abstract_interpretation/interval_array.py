from typing import Sequence

import numpy as np

from abstraqt.abstract_interpretation.interfaces.mixins.equal_abstract_object_nan import EqualAbstractObjectNan
from abstraqt.utils.array.dimensions.representation_dimensions_wrapper import RepresentationDimensionsWrapper
from abstraqt.utils.numpy.lift_to_numpy_array import lift_to_numpy_array
from .interfaces import AbstractArray

atol = 1e-8


class IntervalArray(RepresentationDimensionsWrapper, EqualAbstractObjectNan, AbstractArray):

    ################
    # CONSTRUCTORS #
    ################

    def __init__(self, representation: np.ndarray, check_valid_entries=True):
        representation = lift_to_numpy_array(representation)
        assert isinstance(representation, np.ndarray)

        assert representation.dtype == float
        assert len(representation.shape) >= 1
        assert representation.shape[-1] == 2, f'Invalid shape {representation.shape}'
        if check_valid_entries:
            assert np.all(np.logical_or(
                representation[..., 0] <= representation[..., 1],
                np.all(np.isnan(representation), axis=-1)
            )), f'Invalid representation {representation}'

        AbstractArray.__init__(self, representation)
        RepresentationDimensionsWrapper.__init__(self, 1)

    @staticmethod
    def top(shape: Sequence[int]):
        ret = IntervalArray.empty(shape)
        ret.representation[..., 0] = -np.inf
        ret.representation[..., 1] = np.inf
        return ret

    @classmethod
    def bottom(cls, shape: Sequence[int]):
        ret = cls.empty(shape)
        ret.set_to_bottom()
        return ret

    @staticmethod
    def empty(shape: Sequence[int]):
        shape = tuple(shape) + (2,)
        representation = np.empty(shape, dtype=float)
        return IntervalArray(representation, check_valid_entries=False)

    @staticmethod
    def lift(x):
        if isinstance(x, IntervalArray):
            return x

        assert isinstance(x, np.ndarray) or np.isscalar(x), f'Got {type(x)} instead of np.ndarray'
        x = np.asarray(x)

        # prepare output
        ret = IntervalArray.empty(x.shape)
        ret.representation[..., 0] = x
        ret.representation[..., 1] = x
        return ret

    ########
    # SIZE #
    ########

    def size_element_wise(self):
        return size_element_wise_interval_array_representation(self.representation)

    ########
    # JOIN #
    ########

    def join(self, other):
        # fmin and fmax ignore nans
        ret = np.fmin(self.representation, other.representation)
        ret[..., 1] = np.fmax(self.representation[..., 1], other.representation[..., 1])
        return self.__class__(ret)

    #############
    # CANONICAL #
    #############

    def is_bottom(self):
        return np.any(np.isnan(self.representation))

    def is_point(self):
        return np.all(self.representation[..., 0] == self.representation[..., 1])

    def to_canonical(self):
        if self.is_bottom():
            self.set_to_bottom()
        return self

    def set_to_bottom(self):
        self.representation[...] = np.nan

    ##############
    # OPERATIONS #
    ##############

    @classmethod
    def _add(cls, self, other):
        # internal implementation that can also be used by subclasses
        representation = self.representation + other.representation
        ret = cls(representation)
        if self.is_bottom() or other.is_bottom():
            ret.set_to_bottom()
        return ret

    def __add__(self, other):
        return self._add(self, other)

    def __mul__(self, other):
        other = IntervalArray.lift(other)
        representation = mul__interval_array_representation(self.representation, other.representation)
        return IntervalArray(representation)

    def __neg__(self):
        ret = neg__interval_array_representation(self.representation)
        ret = IntervalArray(ret)
        return ret

    def cos(self):
        representation = cos__interval_array_representation(self.representation)
        return IntervalArray(representation)

    def sum(self, axis=None):
        if axis is not None:
            raise NotImplementedError()

        n_dims = len(self.representation.shape)
        axis = tuple(range(n_dims - 1))

        ret = np.sum(self.representation, axis=axis)
        return IntervalArray(ret)

    ###########
    # HELPERS #
    ###########

    def is_super_set_of(self, other):
        assert isinstance(other, IntervalArray)
        return is_super_set_of__interval_array_representation(self.representation, other.representation)

    def equal_abstract_object(self, other) -> bool:
        return np.all(np.logical_or(
            np.isclose(self.representation, other.representation),
            np.logical_and(
                np.isnan(self.representation),
                np.isnan(other.representation)
            )
        ))

    def get_corners_single(self):
        assert self.representation.shape == (2,)
        if not self.is_bottom():
            yield np.asarray(self.representation[0])
            if self.representation[0] != self.representation[1]:
                yield np.asarray(self.representation[1])

    @staticmethod
    def empty_reference(shape: Sequence[int]):
        return np.empty(shape, dtype=float)

    @staticmethod
    def repr_single(v):
        return f'[{v[0]}, {v[1]}]'


########
# SIZE #
########


def _project_to_unit(x):
    """
    Projects x ∈ [0, ∞] to [0, 1]
    """
    ret = np.arctan(x) / (np.pi / 2)
    ret = np.asarray(ret)
    return ret


def _if_nan_then_zero(x):
    nans = np.isnan(x)
    nans = np.asarray(nans)
    if np.any(nans):
        x[...] = 0
    return x


def size_element_wise_interval_array_representation(representation: np.ndarray) -> np.ndarray:
    ret = representation[..., 0] != representation[..., 1]
    ret = np.asarray(ret, dtype=float)
    if np.any(np.isnan(representation)):
        ret[...] = 0
    return ret


def is_super_set_of_element_wise__interval_array_representation(super_representation: np.ndarray,
                                                                sub_representation: np.ndarray):
    ret = np.logical_and(
        super_representation[..., 0] <= sub_representation[..., 0] + atol,
        super_representation[..., 1] >= sub_representation[..., 1] - atol,
    )
    ret = np.asarray(ret)
    if np.any(np.isnan(sub_representation)):
        ret = np.full_like(ret, fill_value=True)
    return ret


def is_super_set_of__interval_array_representation(super_representation: np.ndarray, sub_representation: np.ndarray):
    super_set = is_super_set_of_element_wise__interval_array_representation(super_representation, sub_representation)
    return np.all(super_set)


##############
# OPERATIONS #
##############


def mul__interval_array_representation(x_representation: np.ndarray, y_representation: np.ndarray):
    lower = np.minimum(
        np.minimum(
            x_representation[..., 0] * y_representation[..., 0],
            x_representation[..., 0] * y_representation[..., 1]
        ),
        np.minimum(
            x_representation[..., 1] * y_representation[..., 0],
            x_representation[..., 1] * y_representation[..., 1]
        )
    )
    upper = np.maximum(
        np.maximum(
            x_representation[..., 0] * y_representation[..., 0],
            x_representation[..., 0] * y_representation[..., 1]
        ),
        np.maximum(
            x_representation[..., 1] * y_representation[..., 0],
            x_representation[..., 1] * y_representation[..., 1]
        )
    )

    # combine
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    ret = np.stack((lower, upper), axis=-1)

    return ret


def cos__interval_array_representation(representation: np.ndarray):
    # handle bottom
    if np.any(np.isnan(representation)):
        return np.full_like(representation, fill_value=np.nan)

    # handle end points
    cos = np.cos(representation)
    lower = np.minimum(cos[..., 0], cos[..., 1])
    upper = np.maximum(cos[..., 0], cos[..., 1])

    # ensure that indexing is possible
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    # prepare for handling extrema
    shifted_phi_intervals = ensure_cyclic_interval_start_in_range(representation)  # starts in range (0, 2π+atol]

    # handle -1
    minus_one_included = np.logical_or(
        np.logical_and(
            shifted_phi_intervals[..., 0] <= np.pi + atol,
            np.pi <= shifted_phi_intervals[..., 1] + atol
        ),
        np.logical_and(
            shifted_phi_intervals[..., 0] <= 3 * np.pi + atol,
            3 * np.pi <= shifted_phi_intervals[..., 1] + atol
        )
    )
    minus_one_included = np.asarray(minus_one_included)
    lower = np.where(minus_one_included, np.full_like(lower, -1), lower)

    # handle 1
    one_included = np.logical_and(
        shifted_phi_intervals[..., 0] <= 2 * np.pi + atol,
        2 * np.pi <= shifted_phi_intervals[..., 1] + atol
    )
    one_included = np.asarray(one_included)
    upper = np.where(one_included, np.full_like(upper, 1), upper)

    # combine
    ret = np.stack((lower, upper), axis=-1)

    return ret


def neg__interval_array_representation(representation: np.ndarray):
    representation = -representation
    representation[..., [0, 1]] = representation[..., [1, 0]]
    return representation


def ensure_cyclic_interval_start_in_range(representation: np.ndarray, cycle_length=2 * np.pi):
    """
    Return a shifted array for which the lower bound starts in (0, cycle_length + atol]
    """
    if np.any(np.isnan(representation)):
        return np.full_like(representation, fill_value=np.nan)

    # subtracting atol moves us to cycle_length even if we are slightly above 0
    offset = (representation[..., 0] - atol) // cycle_length
    offset = np.asarray(offset)
    offset = np.expand_dims(offset, axis=-1)  # broadcast across last axis
    ret = representation - cycle_length * offset

    return ret
