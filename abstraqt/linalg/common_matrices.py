from typing import Union

import numpy as np

############
# MATRICES #
############

I = np.array(((1, 0), (0, 1)), dtype=complex)
X = np.array(((0, 1), (1, 0)), dtype=complex)  # eigenvector for 1: |+〉, eigenvector for -1: |-〉
Y = np.array(((0, -1j), (1j, 0)), dtype=complex)  # eigenvector for 1: |0〉+i|1〉, eigenvector for -1: |0〉-i|1〉
Z = np.array(((1, 0), (0, -1)), dtype=complex)  # eigenvector for 1: |0〉, eigenvector for -1: |1〉
H = 1 / np.sqrt(2) * np.array(((1, 1), (1, -1)), dtype=complex)
S = np.array(((1, 0), (0, 1j)), dtype=complex)
SDG = S.conj().T
T = np.array(((1, 0), (0, np.exp(1j * np.pi / 4))), dtype=complex)  # (0.85 + 0.35i) I + (0.15 - 0.35i) Z
TDG = T.conj().T

projector_to_0 = np.zeros((2, 2))
projector_to_0[0, 0] = 1
projector_to_1 = np.zeros((2, 2))
projector_to_1[1, 1] = 1
CNOT = np.kron(projector_to_0, I) + np.kron(projector_to_1, X)
NOTC = np.kron(I, projector_to_0) + np.kron(X, projector_to_1)
CCNOT = \
    np.kron(projector_to_0, np.kron(projector_to_0, I)) + np.kron(projector_to_0, np.kron(projector_to_1, I)) + \
    np.kron(projector_to_1, np.kron(projector_to_0, I)) + np.kron(projector_to_1, np.kron(projector_to_1, X))
CZ = np.kron(projector_to_0, I) + np.kron(projector_to_1, Z)

paulis_dict = {
    'I': I,
    'X': X,
    'Y': Y,
    'Z': Z,
}
matrices_2x2_dict = {
    **paulis_dict,
    'H': H,
    'S': S,
    'T': T,
    'TDG': TDG,
    'SDG': SDG
}
matrices_all_dict = {
    **matrices_2x2_dict,
    'CNOT': CNOT,
    'NOTC': NOTC,
    'CCNOT': CCNOT,
    'CZ': CZ
}


def extend_by_identities(before: int, m: Union[np.ndarray, str], n: int):
    # before
    ret = np.eye(1 << before)

    # operation itself
    if isinstance(m, str):
        m = matrices_all_dict[m]
    ret = np.kron(ret, m)

    # after
    n_qubits_in_m = int(np.log2(m.shape[0]))
    after = n - before - n_qubits_in_m
    assert after >= 0
    ret = np.kron(ret, np.eye(1 << after))

    return ret


def get_n_qubits(matrix: Union[str, np.ndarray]):
    if isinstance(matrix, str):
        matrix = matrices_all_dict[matrix]
    n = int(np.log2(matrix.shape[0]))
    return n
