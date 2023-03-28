from operator import __mul__, __eq__
from typing import Sequence

import numpy as np

from abstraqt.abstract_interpretation.abstract_bool_array import AbstractBoolArray, invert_abstract_bool_representation, \
    sum_abstract_bool_array_representation
from abstraqt.abstract_interpretation.abstract_int2_array import sum__abstract_int2_array_representation, \
    AbstractInt2Array
from abstraqt.abstract_interpretation.interfaces.abstract_bit_pattern_array import AbstractBitPatternArray, \
    bit_pattern_to_str, join_abstract_bit_pattern_array__representation
from abstraqt.abstract_interpretation.interfaces.best_transformers import add_best_transformer_to_class, \
    from_reference_implementation, \
    get_best_transformer_on_representation
from abstraqt.linalg.common_matrices import get_n_qubits
from abstraqt.utils.numpy.my_numpy import concatenate_last_axis
from .pauli_bit_representation import factor_lookup, PauliBitRepresentation, conjugate_letters_list, \
    commutes_representation_generic, get_conjugate_with_op, get_prefactors_aggregated_generic
from .pauli_reference import pauli_index_to_letter, get_representation
from ..utils.lift.lift import LiftError


class AbstractPauliBitRepresentation(AbstractBitPatternArray):
    """
    Represents abstract elements of the pauli group by f || P,
    where f represents an abstract pre-factor ⊆ {1,i,-1,-i}:
    - 0000 ≘ {}
    - 0001 ≘ {1}
    - 0010 ≘ {i}
    - 0100 ≘ {-1}
    - 1000 ≘ {-i}
    - ...
    - 1111 ≘ {1,i,-1,-i}
    and P represents an abstract element ⊆ {I,X,Y,Z}:
    - 0000 ≘ {}
    - 0001 ≘ {I}
    - 0010 ≘ {X}
    - 0100 ≘ {Y}
    - 1000 ≘ {Z}
    - ...
    - 1111 ≘ {I,X,Y,Z}

    If the dimension is >=1, interprets the last dimension as a tensor product
    of matrices

    Encodes errors as bottom. For example, T @ X @ T† is no Pauli matrix and
    therefore the resulting abstract object would be encoded as bottom
    """

    ################
    # CONSTRUCTORS #
    ################

    def __init__(self, representation: np.ndarray):
        super().__init__(representation)

    @classmethod
    def lift(cls, x):
        if isinstance(x, AbstractPauliBitRepresentation):
            return x
        if isinstance(x, PauliBitRepresentation):
            x = x.representation
        if isinstance(x, np.ndarray):
            ret = AbstractPauliBitRepresentation.lift_concrete_bit_pattern_array(x)
            return AbstractPauliBitRepresentation(ret)
        else:
            raise LiftError(x, AbstractPauliBitRepresentation)

    @classmethod
    def lift_concrete_bit_pattern_array(cls, x: np.ndarray):
        pre_factor = x // 4
        pauli_representation = np.bitwise_and(x, 3)

        ret = (1 << pre_factor) << 4
        ret += (1 << pauli_representation)
        ret = np.asarray(ret, dtype=np.uint8)
        return ret

    @staticmethod
    def identity(*shape: int):
        i = PauliBitRepresentation.identity(*shape)
        ret = AbstractPauliBitRepresentation.lift(i)
        return ret

    ##############
    # OPERATIONS #
    ##############

    bound_on_representation = 1 << 8

    def __mul__(self, other):
        if isinstance(other, PauliBitRepresentation):
            other = AbstractPauliBitRepresentation.lift(other)
        return AbstractPauliBitRepresentation.mul_abstract_only(self, other)

    def __rmul__(self, other):
        if isinstance(other, PauliBitRepresentation):
            other = AbstractPauliBitRepresentation.lift(other)
        return other * self

    def commutes(self, other, all_pairs=False):
        other = AbstractPauliBitRepresentation.lift(other)
        ret = commutes_abstract_pauli_bit_representation(self.representation, other.representation, all_pairs=all_pairs)
        return AbstractBoolArray(ret)

    def conjugate_with(self, op: str, *qubits: int):
        conjugate_with_op = getattr(AbstractPauliBitRepresentation, 'conjugate_with_' + op)
        return conjugate_with_op(self, *qubits)

    ###########
    # CORNERS #
    ###########

    def get_corners_single(self):
        assert self.representation.shape == ()
        for pre_factor in range(4):
            if np.bitwise_and(self.representation, (1 << pre_factor) << 4):
                for pauli_index in range(4):
                    if np.bitwise_and(self.representation, 1 << pauli_index):
                        ret = self.empty_reference(())
                        ret[()] = (pre_factor << 2) + pauli_index
                        yield ret

    @staticmethod
    def empty_reference(shape: Sequence[int]):
        return PauliBitRepresentation(np.empty(shape, dtype=np.uint8))

    ###############
    # CONVENIENCE #
    ###############

    @classmethod
    def repr_single(cls, v):
        pre_factor = v // 16
        pre_factor = bit_pattern_to_str(pre_factor, 4, value_to_str=lambda x: factor_lookup[x])

        pauli_representation = v % 16
        pauli_representation = bit_pattern_to_str(pauli_representation, 4,
                                                  value_to_str=lambda x: pauli_index_to_letter[x])

        return pre_factor + pauli_representation

    def to_canonical(self):
        if len(self.representation.shape) >= 1:
            self.representation = to_canonical_representation(self.representation)
        return self


pauli_mul__abstract_representation = add_best_transformer_to_class(
    AbstractPauliBitRepresentation,
    'mul_abstract_only',
    __mul__
)
eq__abstract_pauli_bit_representation_representation = add_best_transformer_to_class(
    AbstractPauliBitRepresentation,
    '__eq__',
    __eq__,
    target_class=AbstractBoolArray
)


def bool_array_to_pauli_representation(bool_array):
    return get_representation(np.power(-1, bool_array.representation) * np.eye(2, dtype=complex))


add_best_transformer_to_class(
    AbstractBoolArray,
    'to_abstract_pauli_bit_representation',
    bool_array_to_pauli_representation,
    target_class=AbstractPauliBitRepresentation
)


#############
# CONJUGATE #
#############


def add_conjugates():
    for letter in conjugate_letters_list:
        n_qubits = get_n_qubits(letter)
        label = 'conjugate_with_' + letter

        direct = label + '_direct'
        conjugate_with_op_direct_reference = getattr(PauliBitRepresentation, direct)
        conjugate_with_op_direct = get_best_transformer_on_representation(
            AbstractPauliBitRepresentation,
            direct,
            conjugate_with_op_direct_reference,
            target_shape=(n_qubits,)
        )
        conjugate_with_op = get_conjugate_with_op(conjugate_with_op_direct, letter, AbstractPauliBitRepresentation)
        setattr(AbstractPauliBitRepresentation, label, conjugate_with_op)


add_conjugates()

###########
# COMMUTE #
###########


commutes_element_wise = from_reference_implementation(
    AbstractPauliBitRepresentation,
    PauliBitRepresentation.commutes,
    'commutes_abstract_pauli',
    target_class=AbstractBoolArray
)

commutes_abstract_pauli_bit_representation = commutes_representation_generic(
    commutes_element_wise,
    invert_abstract_bool_representation,
    sum_abstract_bool_array_representation,
    np.uint8
)


#############
# CANONCIAL #
#############

def prefactor_and_bare__abstract_pauli_bit_representation(representation: np.ndarray):
    # helper values needed for numba
    fifteen = np.array(15, dtype=np.uint8)
    sixteen = np.array(16, dtype=np.uint8)

    # handle bottom
    pauli_bottom = np.any(np.bitwise_and(representation, fifteen) == 0)
    prefactor_bottom = np.any(np.bitwise_and(representation // sixteen, fifteen) == 0)
    if pauli_bottom or prefactor_bottom:
        representation.fill(0)
        return representation[..., 0], representation

    # handle pre-factors
    pre_factors = representation // sixteen
    sums = sum__abstract_int2_array_representation(pre_factors, axis=-1)
    representation = np.bitwise_and(representation, fifteen)
    representation[..., :] += sixteen
    return sums, representation


def to_canonical_representation(representation: np.ndarray):
    prefactor, representation = prefactor_and_bare__abstract_pauli_bit_representation(representation)
    representation[..., 0] = np.bitwise_or(representation[..., 0], prefactor * 16)
    return representation


####################
# BARE / PREFACTOR #
####################


get_bare_pauli_bitwise__abstract_representation = add_best_transformer_to_class(
    AbstractPauliBitRepresentation,
    'get_bare_pauli_bitwise',
    PauliBitRepresentation.get_bare_pauli_bitwise,
    target_class=AbstractBoolArray,
    target_shape=(2,)
)

get_prefactors_element_wise__abstract_pauli_bit_representation_representation = add_best_transformer_to_class(
    AbstractPauliBitRepresentation,
    'get_prefactors',
    PauliBitRepresentation.get_prefactors,
    target_class=AbstractInt2Array
)

get_prefactors_aggregated__abstract_pauli_bit_representation_representation = get_prefactors_aggregated_generic(
    get_prefactors_element_wise__abstract_pauli_bit_representation_representation,
    sum__abstract_int2_array_representation
)


def get_bare_pauli_concatenated__abstract_representation(representation: np.ndarray):
    representation = np.transpose(representation)
    bare = get_bare_pauli_bitwise__abstract_representation(representation)
    ret = concatenate_last_axis(bare, axis=0)
    return ret


def combine_prefactor_and_bare_pauli__representation(prefactor: np.ndarray, bare: np.ndarray):
    ret = np.bitwise_or(prefactor << 4, bare)
    ret = np.asarray(ret, dtype=np.uint8)
    return ret


##########
# OTHERS #
##########

def select_abstract__pauli_bit_representation_representation(q: np.ndarray, x: np.ndarray):
    """

    - q: concrete pauli bit representation
    - x: abstract bool array
    """
    n, = x.shape
    ret = AbstractPauliBitRepresentation.identity(q.shape[1]).representation
    for i in range(n):
        q_i = AbstractPauliBitRepresentation.lift_concrete_bit_pattern_array(q[i, :])
        if x[i] == AbstractBoolArray.zero_0D__representation:
            pass
        elif x[i] == AbstractBoolArray.one_0D__representation:
            ret = pauli_mul__abstract_representation(ret, q_i)
        else:
            ret_tmp = pauli_mul__abstract_representation(ret, q_i)
            ret = join_abstract_bit_pattern_array__representation(ret, ret_tmp)

    return ret
