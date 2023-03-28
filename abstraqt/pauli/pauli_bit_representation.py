import functools
from operator import __matmul__
from typing import Sequence, Callable

import numpy as np

from abstraqt.linalg.common_matrices import matrices_all_dict, get_n_qubits
from abstraqt.linalg.solve_mod_2 import solve_mod_2
from abstraqt.utils.array.dimensions.representation_wrapper import RepresentationWrapper
from abstraqt.utils.array.dimensions.shaping import pairing
from abstraqt.utils.array.lookup_table import LookupTable
from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.inspection.function_arguments import count_n_arguments
from abstraqt.utils.lift.lift_to_representation import lift_to_representation
from abstraqt.utils.numpy.my_numpy import concatenate_last_axis
from abstraqt.pauli.pauli_reference import pauli_index_to_letter, pauli_representation_from_string, to_matrix, get_representation, \
    matrices_commute, conjugate_with_matrix, to_matrices


class PauliBitRepresentation(RepresentationWrapper):
    """
    Represents elements of the pauli group by f || P,
    where f represents a pre-factor:
    - 00 ≘ 1
    - 01 ≘ i
    - 10 ≘ -1
    - 11 ≘ -i
    and each P represents an element of the 2x2 pauli group:
    - 00 ≘ I
    - 01 ≘ X
    - 10 ≘ Z
    - 11 ≘ Y

    If the dimension is >=1, interprets the last dimension as a tensor product
    of matrices
    """

    ################
    # CONSTRUCTORS #
    ################

    def __init__(self, representation: np.ndarray):
        assert isinstance(representation, np.ndarray), f"Got {type(representation)} instead of np.ndarray"
        assert representation.dtype == np.uint8, f"Got {representation.dtype} instead of uint8"
        self.representation = representation

    @staticmethod
    def from_string(s: str):
        representation = pauli_representation_from_string(s)
        assert len(representation.shape) == 1
        return PauliBitRepresentation(representation)

    @staticmethod
    def identity(*shape: int):
        representation = np.zeros(shape, dtype=np.uint8)
        return PauliBitRepresentation(representation)

    @staticmethod
    def stabilizer_of_zero(n_bits: int):
        representation = 2 * np.eye(n_bits, dtype=np.uint8)
        return PauliBitRepresentation(representation)

    ###########
    # HELPERS #
    ###########

    def to_canonical(self):
        if len(self.representation.shape) >= 1:
            self.representation = to_canonical_representation(self.representation)
        return self

    def encodes_error(self):
        return np.any(self.representation == np.array(-1, dtype=np.uint8))

    ##############
    # OPERATIONS #
    ##############

    def __mul__(self, other):
        if not isinstance(other, PauliBitRepresentation):
            return NotImplemented
        ret = pauli_mul(self.representation, other.representation)
        return PauliBitRepresentation(ret)

    def commutes(self, other, all_pairs=False):
        assert isinstance(other, PauliBitRepresentation)
        return commutes_pauli_bit_representation_representation(self.representation, other.representation,
                                                                all_pairs=all_pairs)

    def conjugate_with(self, op: str, *qubits: int):
        conjugate_with_op = getattr(PauliBitRepresentation, 'conjugate_with_' + op)
        return conjugate_with_op(self, *qubits)

    def solve(self, other):
        assert isinstance(other, PauliBitRepresentation)
        x = solve__pauli_bit_representation_representation(self.representation, other.representation)
        return x

    def select(self, x: np.ndarray):
        result = select__pauli_bit_representation_representation(self.representation, x)
        return PauliBitRepresentation(result)

    def pad_by_identities(self, total_n_bits: int, *qubits: int):
        ret = PauliBitRepresentation.identity(*self.representation.shape[:-1], total_n_bits)
        ret[..., qubits] = self.representation
        return ret

    ###############
    # CONVENIENCE #
    ###############

    def __eq__(self, other):
        if not isinstance(other, PauliBitRepresentation):
            return False
        return np.all(self.representation == other.representation)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(pauli_representation_to_str(self.representation))

    ##################
    # REPRESENTATION #
    ##################

    def to_matrices(self):
        return to_matrices(self.representation)

    def get_bare_pauli(self):
        ret = get_bare_pauli__representation(self.representation)
        return PauliBitRepresentation(ret)

    def get_bare_pauli_bitwise(self):
        return get_bare_pauli_bitwise__representation(self.representation)

    def get_prefactors(self, aggregate=False):
        if aggregate:
            ret = get_prefactors_aggregated__pauli_bit_representation_representation(self.representation)
        else:
            ret = get_prefactors_element_wise__pauli_bit_representation_representation(self.representation)
        return ret


##############
# OPERATIONS #
##############


def to_canonical_representation(representation: np.ndarray):
    four = np.array(4, dtype=np.uint8)  # needed for numba
    overall_pre_factor = np.sum(representation // four, axis=-1, dtype=np.uint8) % four
    overall_pre_factor = np.asarray(overall_pre_factor, dtype=np.uint8)
    representation &= np.asarray(3, dtype=np.uint8)

    overall_pre_factor *= four
    representation[..., 0] = np.bitwise_or(representation[..., 0], overall_pre_factor)
    return representation


def lift_matrix_operation_to_representation(op, output_matrix_to_representation=False, tensor_inputs=False):
    """
    Takes an operation on matrices and lifts it to pauli representation
    """
    lifted_name = op.__name__ + '_lifted'

    @rename_function(lifted_name)
    def lifted(*args):
        try:
            all_args = [to_matrix(a) for a in args]

            if tensor_inputs:
                all_args = [functools.reduce(np.kron, all_args, np.array(1, dtype=complex))]

            result = op(*all_args)
            if output_matrix_to_representation:
                result = get_representation(result)
                result = result.astype(np.uint8)
            return result
        except ValueError:
            # encode error
            return np.array(-1, dtype=np.uint8)

    return lifted


def lookup_table_from_matrix_operation(op, label, output_matrix_to_representation=False, tensor_n_inputs=None):
    reference_implementation = lift_matrix_operation_to_representation(
        op,
        output_matrix_to_representation,
        tensor_inputs=tensor_n_inputs is not None
    )

    if tensor_n_inputs is None:
        n_arguments = count_n_arguments(op)
    else:
        n_arguments = tensor_n_inputs

    table_label = 'Pauli.' + label
    t = LookupTable(reference_implementation, n_arguments * [16], table_label)

    f = t.lookup
    return f


pauli_mul = lookup_table_from_matrix_operation(__matmul__, __matmul__.__name__, output_matrix_to_representation=True)
commutes_element_wise = lookup_table_from_matrix_operation(matrices_commute, 'commutes_pauli')


#########
# SOLVE #
#########


def solve__pauli_bit_representation_representation(a: np.ndarray, b: np.ndarray):
    """
    Returns boolean vector x such that, if a represents S1, ..., Sn and b represents T, then
    T = S1^x1 * ... * Sn^xn   (ignoring pre-factors)
    """
    n, m = a.shape
    assert n == m
    if len(b.shape) <= 1:
        b = np.atleast_2d(b)
    if b.shape[1] != 1:
        b = b.T

    a = get_bare_pauli_bitwise__representation(a.T)
    a = concatenate_last_axis(a, axis=0)
    b = get_bare_pauli_bitwise__representation(b)
    b = concatenate_last_axis(b, axis=0)

    x = solve_mod_2(a, b)
    x = x.flatten()
    return x


def select__pauli_bit_representation_representation(representation: np.ndarray, x: np.ndarray):
    n, = x.shape
    ret = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if x[i]:
            ret = pauli_mul(ret, representation[i, :])
    return ret


####################
# BARE / PREFACTOR #
####################


def get_bare_pauli__representation(representation: np.ndarray):
    return np.bitwise_and(representation, 3)


def get_bare_pauli_bitwise__representation(representation: np.ndarray):
    xs = np.bitwise_and(representation, 1)
    zs = np.bitwise_and(representation // 2, 1)

    xs = np.asarray(xs)
    zs = np.asarray(zs)

    a = np.stack((xs, zs), axis=-1)
    return a


def get_bare_pauli_concatenated__representation(representation: np.ndarray):
    representation = np.transpose(representation)
    bare = get_bare_pauli_bitwise__representation(representation)
    ret = concatenate_last_axis(bare, axis=0)
    return ret


def get_prefactors_aggregated_generic(get_prefactors_element_wise_generic, sum_generic):
    def aggregated(representation: np.ndarray):
        prefactors = get_prefactors_element_wise_generic(representation)
        prefactors = sum_generic(prefactors, axis=-1)
        prefactors = np.asarray(prefactors)
        return prefactors

    return aggregated


def get_prefactors_element_wise__pauli_bit_representation_representation(representation: np.ndarray):
    prefactors = np.bitwise_and(representation // 4, 3)
    prefactors = np.asarray(prefactors)
    return prefactors


def sum_mod_4(x: np.ndarray, axis=None):
    ret = np.sum(x, axis=axis) % 4
    ret = np.asarray(ret, dtype=np.uint8)
    return ret


get_prefactors_aggregated__pauli_bit_representation_representation = get_prefactors_aggregated_generic(
    get_prefactors_element_wise__pauli_bit_representation_representation,
    sum_mod_4
)


############
# COMMUTES #
############


def commutes_representation_generic(commutes_generic, invert_generic, sum_generic, output_dtype):
    def aggregate_commutes(_commutes_element_wise):
        _commutes_element_wise = np.atleast_1d(_commutes_element_wise)  # ensure aggregation is possible for 0d
        anti_commutes_element_wise = invert_generic(_commutes_element_wise)
        # aggregate along last dimension (interpreted as tensor product)
        anti_commutes_element_wise = np.asarray(anti_commutes_element_wise, dtype=np.uint8)
        anti_commutes = sum_generic(anti_commutes_element_wise, axis=-1)
        _commutes = invert_generic(anti_commutes)
        _commutes = np.asarray(_commutes, dtype=output_dtype)
        return _commutes

    def _commutes_representation(p0: np.ndarray, p1: np.ndarray, all_pairs=False):
        if all_pairs:
            p0, p1 = pairing(p0, p1, ignore_last_n_dimensions=1)
        _commutes_element_wise = commutes_generic(p0, p1)
        _commutes = aggregate_commutes(_commutes_element_wise)
        return _commutes

    return _commutes_representation


def sum_mod_2(x, axis=None):
    return np.sum(x, axis=axis, dtype=np.uint8) % 2


commutes_pauli_bit_representation_representation = commutes_representation_generic(
    commutes_element_wise,
    np.logical_not,
    sum_mod_2,
    np.bool_
)


#############
# CONJUGATE #
#############


def get_conjugate_with_op(conjugate_with_op_direct: Callable, letter: str, target_class: type):
    n = get_n_qubits(letter)

    @rename_function('conjugate_with_' + letter)
    def conjugate_with_op(self, *qubits: int):
        old_representation = self.representation

        n_qubits = len(qubits)
        assert n_qubits == n, f'Mismatch between actual ({n_qubits}) and expected ({n}) number of qubits'

        new_representation = old_representation.copy()
        args = tuple(new_representation[..., q] for q in qubits)
        rhs = conjugate_with_op_direct(*args)

        if len(qubits) == 1:
            rhs = np.expand_dims(rhs, axis=-1)

        new_representation[..., qubits] = rhs

        return target_class(new_representation)

    return conjugate_with_op


def add_conjugates(letters: Sequence[str]):
    for letter in letters:
        label = 'conjugate_with_' + letter
        op = conjugate_with_matrix(letter)
        n = get_n_qubits(letter)

        conjugate_with_op_direct = lookup_table_from_matrix_operation(op, label, output_matrix_to_representation=True,
                                                                      tensor_n_inputs=n)
        setattr(PauliBitRepresentation, label + '_direct',
                lift_to_representation(PauliBitRepresentation, conjugate_with_op_direct))

        conjugate_with_op = get_conjugate_with_op(conjugate_with_op_direct, letter, PauliBitRepresentation)
        setattr(PauliBitRepresentation, label, conjugate_with_op)


conjugate_letters_list = matrices_all_dict.keys()
add_conjugates(conjugate_letters_list)

#####################
# TO STRING HELPERS #
#####################


factor_lookup = {
    0: '1',
    1: 'i',
    2: '-',
    3: '-i'
}


def pauli_representation_to_str_single(v):
    pre_factor = v // 4
    pauli = v % 4
    try:
        ret = factor_lookup[pre_factor] + pauli_index_to_letter[pauli]
    except KeyError:
        ret = str(v) + ' (error)'
    return np.array(ret, dtype='<U20')


pauli_representation_to_str = LookupTable(
    pauli_representation_to_str_single,
    [1 << 8],
    'pauli_representation_to_str'
)
