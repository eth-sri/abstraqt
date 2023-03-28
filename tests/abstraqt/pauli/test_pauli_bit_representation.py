import unittest

import numpy as np

from abstraqt.linalg.common_matrices import get_n_qubits, extend_by_identities
from abstraqt.pauli import PauliBitRepresentation, to_matrix, matrices_commute, get_representation
from abstraqt.pauli.pauli_bit_representation import conjugate_letters_list
from tests.abstraqt.pauli.test_pauli_reference import get_random_pauli_bit_representation
from tests.test_config import default_random_repetitions


class TestPauliBitRepresentation(unittest.TestCase):

    def test_mul(self):
        for n in [1, 2, 3]:
            for seed in range(default_random_repetitions):
                p1 = random_pauli_bit_representation_1d(n)
                p2 = random_pauli_bit_representation_1d(n)
                with self.subTest(p1=p1, p2=p2, n=n, seed=seed):
                    p = p1 * p2
                    actual = to_matrix(p.representation)

                    expected = to_matrix(p1.representation) @ to_matrix(p2.representation)
                    np.testing.assert_allclose(actual, expected)

    def test_commutes(self):
        for n in [1, 2, 3]:
            for seed in range(default_random_repetitions):
                p1 = random_pauli_bit_representation_1d(n)
                p2 = random_pauli_bit_representation_1d(n)
                with self.subTest(p1=p1, p2=p2, seed=seed, n=n):
                    actual = p1.commutes(p2)
                    expected = matrices_commute(to_matrix(p1.representation), to_matrix(p2.representation))

                    self.assertEqual(actual, expected)

    def test_conjugate(self):
        for seed in range(default_random_repetitions):
            for letter in conjugate_letters_list:
                for n in [1, 2]:
                    self.check_conjugate(seed, letter, n)

    def check_conjugate(self, seed, letter, n):
        np.random.seed(seed)
        p = random_pauli_bit_representation_1d(n)
        n_qubits_for_op = get_n_qubits(letter)

        if n_qubits_for_op > n:
            return

        with self.subTest(p=p, letter=letter, seed=seed, n=n):
            # compute actual
            function_name = 'conjugate_with_' + letter
            op = getattr(PauliBitRepresentation, function_name)
            actual = op(p, *range(n_qubits_for_op))

            # compute expected
            m = to_matrix(p.representation)
            c = extend_by_identities(0, letter, n)
            expected = c @ m @ c.conjugate().transpose()

            try:
                # check if result is pauli
                _ = get_representation(expected)
                # compare
                actual_matrix = to_matrix(actual.representation)
                np.testing.assert_allclose(actual_matrix, expected, atol=1e-7)
            except ValueError:
                self.assertTrue(actual.encodes_error())

    def test_solve_zeros(self):
        s = PauliBitRepresentation.stabilizer_of_zero(3)
        goal = PauliBitRepresentation(np.zeros((1, 3), dtype=np.uint8))
        x = s.solve(goal)
        np.testing.assert_equal(x, 0)

    def test_solve(self):
        for n in [2, 3]:
            for seed in range(default_random_repetitions):
                np.random.seed(seed)
                s = get_random_pauli_bit_representation(n)
                s = PauliBitRepresentation(s)
                with self.subTest(n=n, seed=seed, s=s):
                    expected_x = np.zeros(n)
                    expected_x[0] = 1
                    expected_x[1] = 1

                    goal = s.select(expected_x)
                    x = s.solve(goal)

                    np.testing.assert_equal(x, expected_x)


def random_pauli_bit_representation_1d(n: int):
    representation = np.random.randint(0, 8, size=(n,), dtype=np.uint8)
    return PauliBitRepresentation(representation)
