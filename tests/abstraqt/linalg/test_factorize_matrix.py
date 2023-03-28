from unittest import TestCase

import numpy as np

from abstraqt.linalg.common_matrices import matrices_2x2_dict
from abstraqt.linalg.factorize_matrix import factorize


def get_vectors():
	n = 4
	for index in range(n):
		vector = np.zeros(n)
		vector[index] = 1
		yield vector
	yield np.zeros(n)


def get_matrices():
	n = 4
	yield np.identity(n)
	yield np.zeros((n, n))
	matrix = np.zeros((n, n))
	for i in range(0, n, 2):
		matrix[i, i] = 2
	yield matrix
	for matrix1 in matrices_2x2_dict.values():
		for matrix2 in matrices_2x2_dict.values():
			yield np.kron(matrix1, matrix2)


class FactorizeTest(TestCase):

	def test_factorize_vectors(self):
		for vector in get_vectors():
			with self.subTest(vector=vector):
				a, b = factorize(vector, [2])
				self.assertIsNotNone(a)
				self.assertIsNotNone(b)

				prod = np.kron(a, b)
				np.testing.assert_almost_equal(vector, prod)

	def test_factorize_matrices(self):
		for matrix in get_matrices():
			with self.subTest(matrix=matrix):
				a, b = factorize(matrix, [2, 2])
				self.assertIsNotNone(a)
				self.assertIsNotNone(b)

				prod = np.kron(a, b)
				np.testing.assert_almost_equal(matrix, prod)
