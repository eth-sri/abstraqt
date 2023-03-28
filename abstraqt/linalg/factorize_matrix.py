from typing import Tuple, List

import numpy as np


def factorize(matrix: np.ndarray, shape_a: List[int]):
	"""
	Returns matrices A, B with:
	- A.shape = shape_a
	- A ⊗ B = matrix
	If this is impossible, returns None, None
	"""
	assert len(matrix.shape) == len(shape_a)

	# get shape of b
	shape_b = [matrix.shape[i] // shape_a[i] for i in range(len(shape_a))]
	for i in range(len(shape_a)):
		assert shape_a[i] * shape_b[i] == matrix.shape[i]

	# prepare matrices
	a = np.zeros(shape_a, dtype=matrix.dtype)
	b = np.zeros(shape_b, dtype=matrix.dtype)

	# special case for all zeros
	if not matrix.any():
		return a, b

	# find non-zero value
	indices = np.argwhere(np.abs(matrix) > 1e-8)[0]
	indices_a = tuple(indices[i] // shape_b[i] for i in range(len(shape_b)))
	# assume w.l.o.g. that a is one there
	a[indices_a] = 1
	# this fixes b
	b = get_block(matrix, indices_a, shape_b)
	# fill out the rest consistently
	for indices_a in np.ndindex(a.shape):
		if a[indices_a] == 0:
			scaled_b = get_block(matrix, indices_a, shape_b)
			factor = get_factor(b, scaled_b)
			if factor is None:
				return None, None
			else:
				a[indices_a] = factor

	return a, b


def get_factor(matrix1: np.ndarray, matrix2: np.ndarray):
	"""
	a such that a*matrix1 == matrix2
	"""
	if not matrix1.shape == matrix2.shape:
		return None

	# move values close to zero to zero (avoids numerical issues)
	matrix1[np.isclose(matrix1, 0)] = 0
	matrix2[np.isclose(matrix2, 0)] = 0

	if not matrix2.any():
		# all zeros
		return 0

	with np.errstate(divide='ignore', invalid='ignore'):
		candidates = matrix2 / matrix1
	# error if 1/0
	if np.isinf(candidates).any():
		return None
	# ignore 0/0
	candidates = candidates[~np.isnan(candidates)]
	# extract very first entry (accounts for unknown dimension of input)
	# Note: the must be at least one candidate, because matrix2 != 0
	val = candidates[tuple(0 for _ in range(len(candidates.shape)))]
	# error on conflicting candidates
	if not np.allclose(val, candidates):
		return None
	return val


def get_block(matrix: np.ndarray, indices_a: Tuple, shape_b: List):
	slices = [slice(
		shape_b[i] * indices_a[i],
		shape_b[i] * (indices_a[i] + 1)
	) for i in range(len(shape_b))]
	slices = tuple(slices)
	return matrix[slices]


def factorize_into_2x2(matrix: np.ndarray):
	"""
	Returns a list [A1, ..., An] with
	- A1 ⊗ ... ⊗ An = matrix
	- Ai.shape = (2,2)
	If this is impossible, returns None
	"""
	assert len(matrix.shape) == 2
	if matrix.shape == (2, 2):
		return [matrix]
	else:
		a, b = factorize(matrix, [2, 2])
		if a is None:
			return None
		return [a] + factorize_into_2x2(b)
