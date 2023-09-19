import numpy as np

from abstraqt.linalg.solve_mod_2.solve_helper import solve_via_gaussian_elimination, \
    kernel_via_gaussian_elimination
from abstraqt.utils.my_numpy.my_numba import my_njit, outer_with_and
from abstraqt.utils.my_numpy.pack_bits import unpack_bits, pack_bits


@my_njit
def gaussian_elimination_mod_2(a: np.ndarray, full_gaussian=True):
    n_rows, n_columns = a.shape

    u = a

    # implementation loosely based on https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
    pivot_row = 0
    pivot_column = 0

    while pivot_row < n_rows and pivot_column < n_columns:
        # find next pivot
        good_row = int(u[pivot_row:, pivot_column].argmax()) + pivot_row
        # good_row = pivot_row
        if u[good_row, pivot_column] == 0:
            # no pivot this column, pass to next column
            pivot_column += 1
        else:
            # swap rows
            swap_rows(u, good_row, pivot_row)

            # normalize pivot row
            pivot = u[pivot_row, pivot_column]
            assert pivot != 0

            if pivot_row < n_rows:
                rows_list = [slice(pivot_row + 1, n_rows)]

                # also update block above current pivot if full_gaussian
                if full_gaussian:
                    if pivot_row > 0:
                        rows_list += [slice(0, pivot_row)]

                # set pivot to 1 (not needed for boolean arrays)
                # u[pivot_row, :] /= pivot
                # if return_l:
                #     l[:, pivot_row] *= pivot

                # handle linear combinations
                for rows in rows_list:
                    columns = slice(pivot_column, n_columns)
                    factors_per_row = u[rows, pivot_column]
                    pivot_row_data = u[pivot_row, columns]
                    addition = np.outer(factors_per_row, pivot_row_data)
                    u[rows, columns] ^= addition

            # increase pivot row and column
            pivot_row += 1
            pivot_column += 1

    return u


n_packed = 32
packed_dtype = np.uint32


@my_njit
def gaussian_elimination_mod_2_packed_helper(a: np.ndarray, n_columns_unpacked: int, full_gaussian=True):
    # assert a.dtype == packed_dtype
    zero = np.array(0, dtype=packed_dtype)
    one = np.array(1, dtype=packed_dtype)

    n_rows, n_columns = a.shape
    assert n_packed * (n_columns - 1) < n_columns_unpacked <= n_packed * n_columns

    u = a

    # implementation loosely based on https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
    pivot_row = 0
    pivot_column = 0

    while pivot_row < n_rows and pivot_column < n_columns_unpacked:
        # handle bitpacking
        pivot_column_index = pivot_column // n_packed
        pivot_column_offset = pivot_column % n_packed
        pivot_column_offset_from_right = n_packed - pivot_column_offset - 1
        threshold_for_one = 1 << pivot_column_offset_from_right  # threshold that ensures the entry is one

        # find next pivot
        good_rows = u[pivot_row:, pivot_column_index] >= threshold_for_one
        good_row = int(good_rows.argmax()) + pivot_row
        # good_row = pivot_row
        if u[good_row, pivot_column_index] < threshold_for_one:
            # no pivot this column, pass to next column
            pivot_column += 1
        else:
            # swap rows
            swap_rows(u, good_row, pivot_row)

            # normalize pivot row
            pivot = u[pivot_row, pivot_column_index]
            assert pivot >= threshold_for_one

            if pivot_row < n_rows:
                rows_list = [slice(pivot_row + 1, n_rows)]

                # also update block above current pivot if full_gaussian
                if full_gaussian:
                    if pivot_row > 0:
                        rows_list += [slice(0, pivot_row)]

                # set pivot to 1 (not needed for boolean arrays)
                # u[pivot_row, :] /= pivot
                # if return_l:
                #     l[:, pivot_row] *= pivot

                # handle linear combinations
                for rows in rows_list:
                    columns = slice(pivot_column_index, n_columns)
                    operate_on_rows = (u[rows, pivot_column_index] >> pivot_column_offset_from_right) & one
                    factors_per_row = zero - operate_on_rows
                    pivot_row_data = u[pivot_row, columns]
                    addition = outer_with_and(factors_per_row.astype(packed_dtype), pivot_row_data.astype(packed_dtype))
                    u[rows, columns] ^= addition

            # increase pivot row and column
            pivot_row += 1
            pivot_column += 1

    return u


@my_njit
def gaussian_elimination_mod_2_packing(a: np.ndarray, full_gaussian=True):
    dtype = a.dtype

    # prepare
    n_rows, n_columns = a.shape

    # pad
    missing = (-n_columns) % n_packed
    padding = np.zeros((n_rows, missing), dtype=dtype)
    a_padded = np.hstack((a, padding))

    # reshape and pack
    n_columns_padded = n_columns + missing
    n_columns_packed = n_columns_padded // n_packed
    a_padded = np.reshape(a_padded, (n_rows, n_columns_packed, n_packed))
    a_packed = pack_bits(a_padded, dtype=packed_dtype)

    # solve
    ret_packed = gaussian_elimination_mod_2_packed_helper(a_packed, n_columns, full_gaussian=full_gaussian)

    # unpack
    ret = unpack_bits(ret_packed, n=n_packed)
    ret = np.reshape(ret, (n_rows, n_columns_padded))
    ret = ret[:n_rows, :n_columns]

    ret = ret.astype(dtype)
    return ret


# WARNING: Do not try to implement pseudo-inverse.
#
# This does not work in GF2.
#
# For example,
# [[1 1]] [[x1] [x2]] = [1]
# yields a pseudo-inverse of 0


solve_mod_2_gaussian_elimination = solve_via_gaussian_elimination(gaussian_elimination_mod_2)
get_kernel_mod_2_gaussian_elimination = kernel_via_gaussian_elimination(gaussian_elimination_mod_2)

solve_mod_2_gaussian_elimination_packing = solve_via_gaussian_elimination(gaussian_elimination_mod_2_packing)
get_kernel_mod_2_gaussian_elimination_packing = kernel_via_gaussian_elimination(gaussian_elimination_mod_2_packing)


def invert_matrix_mod_2(a: np.ndarray):
    """
    Returns b such that b @ a = I
    """
    n, m = a.shape
    assert n >= m

    a = np.hstack((a, np.eye(n, dtype=a.dtype)))
    a = gaussian_elimination_mod_2(a)
    return a[:, m:]


@my_njit
def swap_rows(a: np.ndarray, row1: int, row2: int):
    row1_data = a[row1, :].copy()
    a[row1, :] = a[row2, :]
    a[row2, :] = row1_data
# u[[pivot_row, good_row], :] = u[[good_row, pivot_row], :]
