import functools
import itertools
from typing import Sequence, Union

import numpy as np

from abstraqt.linalg.common_matrices import paulis_dict, matrices_all_dict
from abstraqt.utils.function_renamer import rename_function

###########
# HELPERS #
###########


pauli_index_to_letter = {
    0: 'I',
    1: 'X',
    2: 'Z',
    3: 'Y'
}
pauli_letter_to_index = {
    letter: index for index, letter in pauli_index_to_letter.items()
}
pauli_index_to_matrix = {
    index: paulis_dict[letter] for index, letter in pauli_index_to_letter.items()
}
factor_index_to_value = {
    0: 1,
    1: 1j,
    2: -1,
    3: -1j
}


################################
# CONVERSION TO/FROM REFERENCE #
################################


def get_representation(matrix: np.ndarray):
    """

    :param matrix:
    :return: a bit encoding of the matrix
    """
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == complex
    # check matrix size
    n = int(np.log2(matrix.shape[0]))
    assert np.power(2, n) == matrix.shape[0]

    for representation_val in itertools.product(range(16), repeat=n):
        representation = np.array(representation_val, dtype=np.uint8)
        candidate_matrix = to_matrix(representation)
        if np.allclose(candidate_matrix, matrix):
            return representation

    raise ValueError('Matrix is no element from the pauli group')


def to_matrix(representation: np.ndarray):
    assert isinstance(representation, np.ndarray), f'Got {type(representation)} instead of np.ndarray'
    representation = np.atleast_1d(representation)
    assert len(representation.shape) == 1
    # product of matrices
    matrices = [factor_index_to_value[r >> 2] * pauli_index_to_matrix[r & 3] for r in representation]
    matrix = functools.reduce(np.kron, matrices, np.array(1, dtype=complex))
    return matrix


def to_matrices(representation: np.ndarray):
    assert isinstance(representation, np.ndarray)

    n = representation.shape[-1]
    if 0 in representation.shape:
        # cannot apply_along_axis when any iteration dimensions are 0
        return np.zeros(representation.shape[:-1] + (1 << n, 1 << n), dtype=complex)
    else:
        matrices = np.apply_along_axis(to_matrix, axis=-1, arr=representation)
        return matrices


def pauli_representation_from_string(s: str):
    # handle sign
    pauli_string, f = handle_sign(s)

    # get representation
    representation = np.fromiter((pauli_letter_to_index[letter] for letter in pauli_string), dtype=np.uint8)
    representation[0] += f << 2
    return representation


def handle_sign(pauli_string: str):
    # allow signs anywhere in the string
    minus = pauli_string.count('-')
    i = pauli_string.count('i')
    factor = (2 * minus + i) % 4
    pauli_string = pauli_string.replace('-', '')
    pauli_string = pauli_string.replace('+', '')
    pauli_string = pauli_string.replace('i', '')
    return pauli_string, factor


########################
# REFERENCE OPERATIONS #
########################


def matrices_commute(p1: np.ndarray, p2: np.ndarray):
    c = np.allclose(p1 @ p2, p2 @ p1)
    return np.array(c, dtype=bool)


def conjugate_with_matrix(m: Union[str, np.ndarray]):
    if isinstance(m, str):
        matrix = matrices_all_dict[m]
    else:
        matrix = m

    def conjugate(p: np.ndarray):
        ret = matrix @ p @ matrix.conjugate().transpose()
        return ret

    if isinstance(m, str):
        conjugate = rename_function('conjugate_with_' + m)(conjugate)

    return conjugate


def measure_with_matrix(m: np.ndarray, density: np.ndarray):
    op = 1 / 2 * (np.eye(m.shape[0]) + m)
    ret = op @ density @ op
    return ret


def stabilizer_matrices_to_densities(matrices: Sequence[np.ndarray]):
    size = matrices[0].shape[0]
    identity = np.eye(size)
    ret = identity

    for m in matrices:
        mul = (identity + m) / 2
        ret = ret @ mul

    return ret


#############
# DECOMPOSE #
#############

def decompose_into_pauli_basis(a: Union[str, np.ndarray]):
    if isinstance(a, str):
        a = matrices_all_dict[a]

    # check number of qubits
    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1]
    n = int(np.log2(a.shape[0]))

    # prepare options
    string_options = itertools.product(paulis_dict.keys(), repeat=n)
    string_options = [''.join(s) for s in string_options]
    representations = [pauli_representation_from_string(s) for s in string_options]
    matrices = [to_matrix(r) for r in representations]

    a = a.flatten()
    factors = [inner_product_complex(m.flatten() / (1 << n), a) for m in matrices]

    filtered = [(f, r) for f, r in zip(factors, representations) if not np.isclose(f, 0)]

    factors = np.fromiter((f[0] for f in filtered), dtype=complex)
    representations = np.array(list(f[1] for f in filtered), dtype=np.uint8)

    return factors, representations


def inner_product_complex(v1: np.ndarray, v2: np.ndarray):
    return np.inner(np.conj(v1), v2)


def recompose(factors: np.ndarray, representations: np.ndarray, n_bits: int):
    assert factors.shape[0] == representations.shape[0]

    size = 1 << n_bits
    composed = np.zeros((size, size), dtype=complex)
    for factor, pauli_representation in zip(factors, representations):
        composed += factor * to_matrix(pauli_representation)
    return composed
