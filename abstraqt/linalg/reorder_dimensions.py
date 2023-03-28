from typing import Union
import numpy as np

from abstraqt.linalg.common_matrices import matrices_all_dict
from abstraqt.pauli.pauli_reference import handle_sign


def commutation_matrix(m: int, n: int, pad_left=1, pad_right=1):
    """

    :param m:
    :param n:
    :param pad_left:
    :param pad_right:
    :return: Commutation matrix I ⊗ K_{m,n} ⊗ I
    """
    # implementation inspired by https://en.wikipedia.org/wiki/Commutation_matrix
    #
    # Alternative implementation that seems to expose a scipy bug:
    # https://stackoverflow.com/questions/60678746/compute-commutation-matrix-in-numpy-scipy-efficiently

    k = np.zeros((n * m, n * m), dtype=complex)
    for i in range(m):
        for j in range(n):
            k[i + m * j, j + n * i] = 1

    left = np.eye(pad_left, dtype=complex)
    right = np.eye(pad_right, dtype=complex)
    k_padded = np.kron(np.kron(left, k), right)
    return k_padded


def switch_tensor(matrix: np.ndarray):
    k = commutation_matrix(2, 2)
    return k @ matrix @ k


def switch_adjacent_tensor(matrix: np.ndarray, a_length: int = 1, b_length: int = 2, c_length: int = 2):
    """

    :param matrix: A ⊗ B ⊗ C ⊗ D
    :param a_length: A.shape = (a_length, a_length)
    :param b_length:
    :param c_length:
    :return: A ⊗ C ⊗ B ⊗ D
    """
    # https://math.stackexchange.com/questions/3427421/can-permutation-similarities-change-order-of-kronecker-products
    # Kb,c⋅(B⊗C)⋅Kb,c=C⊗B where B is b×b and C is c×c
    n = matrix.shape[0]
    assert matrix.shape == (n, n)

    d_length = n // a_length // b_length // c_length
    assert a_length * b_length * c_length * d_length == n

    k_left = commutation_matrix(b_length, c_length, a_length, d_length).astype(complex)
    k_right = commutation_matrix(c_length, b_length, a_length, d_length).astype(complex)
    ret = k_left @ matrix @ k_right
    return ret


def pad_matrix_by_identities(a: Union[str, np.ndarray], n_total: int, *qubits: int):
    """

    :param a: a sum of summands of the form A1 ⊗ ... ⊗ An
    :param n_total: the total number of qubits to use
    :param qubits:
    :return: a sum of summands of the form I ⊗ A1 ⊗ ... ⊗ An ⊗ I
    such that Ai is at position qubits[i]
    """
    if isinstance(a, str):
        a, factor = handle_sign(a)
        factor = np.power(1j, factor)
        a = factor * matrices_all_dict[a]

    targets = np.array(qubits)

    # sort a
    ordered = False
    while not ordered:
        ordered = True
        for i in range(len(qubits) - 1):
            if targets[i] > targets[i + 1]:
                a = switch_adjacent_tensor(a, 1 << i)
                targets[[i, i + 1]] = targets[[i + 1, i]]
                ordered = False

    n_inserted = 0
    targets = np.append(targets, n_total)
    for i in range(len(targets)):
        n_to_insert = targets[i] - n_inserted
        before = np.eye(1 << n_to_insert, dtype=complex)
        a = np.kron(before, a)
        a = switch_adjacent_tensor(a, 1, 1 << n_to_insert, 1 << n_inserted)
        n_inserted = targets[i] + 1

    return a
