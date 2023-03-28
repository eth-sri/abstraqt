import itertools
from unittest import TestCase

import numpy as np

from abstraqt.linalg.common_matrices import matrices_2x2_dict, I, X, CNOT
from abstraqt.linalg.reorder_dimensions import switch_tensor, switch_adjacent_tensor, commutation_matrix, \
    pad_matrix_by_identities

matrices_to_test = matrices_2x2_dict.values()


class ReorderTest(TestCase):

    def test_switch(self):
        for m1, m2 in itertools.product(matrices_to_test, matrices_to_test):
            with self.subTest(m1=m1, m2=m2):
                prod = np.kron(m1, m2)
                expected = np.kron(m2, m1)
                actual = switch_tensor(prod)
                np.testing.assert_array_almost_equal(expected, actual)

    def test_switch_adjacent_tensor(self):
        a_length = 2
        b_length = 3
        c_length = 5
        d_length = 7
        a = np.random.random((a_length, a_length)).astype(complex)
        b = np.random.random((b_length, b_length)).astype(complex)
        c = np.random.random((c_length, c_length)).astype(complex)
        d = np.random.random((d_length, d_length)).astype(complex)

        matrix = np.kron(np.kron(a, b), np.kron(c, d))

        switched = switch_adjacent_tensor(matrix, a_length, b_length, c_length)

        expected = np.kron(np.kron(a, c), np.kron(b, d))

        self.assertEqual(switched.shape, expected.shape)
        np.testing.assert_allclose(switched, expected)

    @staticmethod
    def test_commutation_matrix():
        # https://en.wikipedia.org/wiki/Commutation_matrix
        a = np.arange(6).reshape(3, 2)
        v2 = a.flatten()
        v1 = a.T.flatten()

        k = commutation_matrix(3, 2)
        np.testing.assert_equal(k.T @ k, np.eye(6))
        np.testing.assert_equal(k @ k.T, np.eye(6))

        np.testing.assert_equal(k.T @ v1, v2)
        np.testing.assert_equal(k @ v2, v1)

    def test_pad_X(self):
        for position in range(2):
            with self.subTest(position=position):
                padded = pad_matrix_by_identities(X, 2, position)

                before = np.eye(1 << position)
                after = np.eye(1 << (1 - position))
                expected = np.kron(np.kron(before, X), after)

                np.testing.assert_equal(padded, expected)

    def test_pad_CNOT(self):
        proj_0 = np.array([[1, 0], [0, 0]])
        proj_1 = np.array([[0, 0], [0, 1]])
        for position_control in range(3):
            for position_target in range(3):
                if position_control == position_target:
                    continue

                position_no_op = {0, 1, 2}
                position_no_op.remove(position_control)
                position_no_op.remove(position_target)
                position_no_op, = position_no_op

                with self.subTest(position_control=position_control, position_target=position_target):
                    padded = pad_matrix_by_identities(CNOT, 3, position_control, position_target)

                    first = [None for _ in range(3)]
                    second = [None for _ in range(3)]

                    first[position_control] = proj_0
                    second[position_control] = proj_1

                    first[position_target] = I
                    second[position_target] = X

                    first[position_no_op] = I
                    second[position_no_op] = I

                    expected = \
                        np.kron(np.kron(first[0], first[1]), first[2]) + \
                        np.kron(np.kron(second[0], second[1]), second[2])

                    np.testing.assert_equal(padded, expected)
