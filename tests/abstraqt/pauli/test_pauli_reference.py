import unittest

import numpy as np

from abstraqt.linalg.common_matrices import paulis_dict, I, X, Y
from abstraqt.utils.my_numpy.pack_bits import pack_bits
from abstraqt.pauli.pauli_bit_representation import commutes_pauli_bit_representation_representation
from abstraqt.pauli.pauli_reference import to_matrices, stabilizer_matrices_to_densities, decompose_into_pauli_basis, \
    recompose
from tests.abstraqt.linalg.random_matrix import get_bool_matrix


def _get_potentially_invalid_random_pauli_bit_representation(n: int):
    representation = get_bool_matrix(n, 2 * n, full_rank=True).astype(np.uint8)
    representation = np.reshape(representation, (n, n, 2))
    signs = np.random.randint(0, 2, size=(n, n, 1)).astype(np.uint8)
    zeros = np.zeros((n, n, 1), dtype=np.uint8)
    representation = np.dstack((signs, zeros, representation))
    representation = np.reshape(representation, (n, n, 4))
    representation = pack_bits(representation, dtype=np.uint8)
    return representation


def get_random_pauli_bit_representation(n: int):
    while True:
        representation = _get_potentially_invalid_random_pauli_bit_representation(n)
        r1 = np.expand_dims(representation, 0)
        r2 = np.expand_dims(representation, 1)
        commutes = commutes_pauli_bit_representation_representation(r1, r2)
        assert commutes.shape == (n, n)
        if np.all(commutes == np.ones(n)):
            return representation


class TestPauliReference(unittest.TestCase):

    def test_to_matrices(self):
        for n in [1, 2, 3]:
            n_rows = np.power(2, n)

            for seed in range(10):
                np.random.seed(seed)

                with self.subTest(n=n, seed=seed):
                    representation = get_random_pauli_bit_representation(n)
                    matrices = to_matrices(representation)
                    for m in matrices:
                        self.assertEqual((n_rows, n_rows), m.shape)

                    density = stabilizer_matrices_to_densities(matrices)
                    self.assertEqual((n_rows, n_rows), density.shape)

                    for m in matrices:
                        conjugated = m @ density @ m.conjugate().transpose()
                        np.testing.assert_allclose(conjugated, density)

    def test_decompose(self):
        for letter, matrix in paulis_dict.items():
            self.check_decompose(letter, matrix)

        custom = 0.4 * I + 1j * X - 0.3j * Y
        self.check_decompose('custom', custom)

    def check_decompose(self, label, matrix):
        with self.subTest(label):
            factors, representations = decompose_into_pauli_basis(matrix)
            n_qubits = int(np.log2(matrix.shape[0]))
            actual = recompose(factors, representations, n_qubits)
            np.testing.assert_almost_equal(actual, matrix)
