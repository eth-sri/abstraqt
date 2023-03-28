from abc import ABC
from typing import Sequence, Dict, Any

import numpy as np

from abstraqt.abstract_interpretation.interfaces import AbstractArray
from abstraqt.utils import logging
from abstraqt.utils.array.lookup_table import LookupTable
from abstraqt.utils.array.mod_array import ModArray
from abstraqt.utils.numpy.lift_to_numpy_array import lift_to_numpy_array
from abstraqt.utils.numpy.my_numpy import bitwise_implies, is_bitwise_one, count_one_bits

logger = logging.getLogger(__name__)


class AbstractBitPatternArray(AbstractArray, ABC):
    """
    Represents abstract elements by bit patterns, where the i-th bit indicates whether the i-th value is possible
    """

    ################
    # CONSTRUCTORS #
    ################

    def __init__(self, representation: np.ndarray):
        super().__init__(representation)
        assert representation.dtype == np.uint8, f"Got {representation.dtype} instead of np.uint8"

    @classmethod
    def top(cls, shape: Sequence[int]):
        return cls(np.full(shape=shape, fill_value=cls.bound_on_representation - 1, dtype=np.uint8))

    @classmethod
    def bottom(cls, shape: Sequence[int]):
        return cls(np.full(shape=shape, fill_value=0, dtype=np.uint8))

    @classmethod
    def empty(cls, shape: Sequence[int]):
        return cls(np.empty(shape=shape, dtype=np.uint8))

    @classmethod
    def lift(cls, x):
        if isinstance(x, cls):
            return x
        if isinstance(x, ModArray):
            x = x.representation
        # prepare output
        ret = cls.lift_concrete_bit_pattern_array(x)
        return cls(ret)

    @classmethod
    def lift_concrete_bit_pattern_array(cls, x):
        x = lift_to_numpy_array(x)
        ret = np.asarray(1 << x, dtype=np.uint8)
        return ret

    #############
    # CONSTANTS #
    #############

    # will be set automatically for every subclass by __init_subclass__
    top_0D__representation = None
    bottom_0D__representation = None

    constants_to_load: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        # general constants
        cls.top_0D__representation = cls.top(()).representation
        cls.bottom_0D__representation = cls.bottom(()).representation

        # specific constants
        for label, value in cls.constants_to_load.items():
            constant = cls.lift(value).representation
            setattr(cls, label + '_0D__representation', constant)

        # load size
        cls.load_size_element_wise()

        # cache repr_element_wise
        cls.cache_repr_element_wise()

    ########
    # SIZE #
    ########

    @classmethod
    def load_size_element_wise(cls):
        """
        Will be called automatically for every subclass by __init_subclass__
        """
        assert cls != AbstractArray

        def size_element_wise_reference(arg):
            max_n_elements = len(list(cls.top(()).get_corners()))
            n_elements = len(list(cls(arg).get_corners()))
            return np.array(n_elements / max_n_elements, dtype=float)

        bounds = [cls.bound_on_representation]
        t = LookupTable(size_element_wise_reference, bounds, cls.__name__ + '.size_element_wise')

        def lookup(self):
            ret = t(self.representation)
            return ret

        setattr(cls, 'size_element_wise', lookup)

    def size_element_wise(self):
        # should be overwritten by "load_size_element_wise"
        raise ValueError("Function load_size_element_wise was not called on class " + self.__class__.__name__)

    ########
    # JOIN #
    ########

    def join(self, other):
        ret = join_abstract_bit_pattern_array__representation(self.representation, other.representation)
        return self.__class__(ret)

    #############
    # CANONICAL #
    #############

    def to_canonical(self):
        if np.any(self.representation == self.bottom_0D__representation):
            self.representation.fill(int(self.bottom_0D__representation))

        return self

    def is_bottom(self):
        if np.any(self.representation == self.bottom_0D__representation):
            return True
        else:
            return False

    def is_point(self):
        ret = count_one_bits(self.representation) == 1
        ret = np.all(ret)
        return ret

    ###########
    # CORNERS #
    ###########

    def get_corners_single(self):
        assert self.representation.shape == () or self.representation.shape == (1,)
        for i in range(self.bound_on_representation):
            if np.bitwise_and(self.representation, 1 << i):
                ret = self.empty_reference(())
                ret[()] = i
                yield ret

    @staticmethod
    def empty_reference(shape: Sequence[int]):
        """
        An empty instance of the reference class
        """
        return np.empty(shape, dtype=np.uint8)

    ############
    # SUPERSET #
    ############

    def is_super_set_of(self, other):
        assert isinstance(other, self.__class__)

        super_set = bitwise_implies(other.representation, self.representation)
        super_set = is_bitwise_one(super_set)

        return np.all(super_set)

    ###############
    # CONVENIENCE #
    ###############

    def repr_element_wise(self):
        return self.__class__.repr_element_wise__representation(self.representation)

    @classmethod
    def repr_element_wise__representation(cls, representation: np.ndarray):
        raise NotImplementedError('Method should have been initialized by cache_repr_element_wise')

    @classmethod
    def repr_single(cls, v):
        ret = bit_pattern_to_str(v, cls.bound_on_representation)
        return ret

    @classmethod
    def cache_repr_element_wise(cls):
        # will be called for each subclass by __init_subclass__

        def reference_repr_single(v):
            ret = cls.repr_single(v)
            ret = np.array(ret, dtype='<U20')
            return ret

        t = LookupTable(reference_repr_single, [cls.bound_on_representation], cls.__name__ + '.__repr_single')
        cls.repr_element_wise__representation = t.lookup
        cls.repr_single = None


def join_abstract_bit_pattern_array__representation(x: np.ndarray, y: np.ndarray):
    ret = np.bitwise_or(x, y)
    ret = np.asarray(ret, dtype=np.uint8)
    return ret


def bit_pattern_to_str(pattern, bound_on_representation, value_to_str=str):
    values = [value_to_str(i) for i in range(bound_on_representation) if np.bitwise_and(pattern, 1 << i)]
    ret = '{' + ','.join(values) + '}'
    return ret
