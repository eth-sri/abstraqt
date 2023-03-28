import numpy as np

from abstraqt.utils.numpy.my_numba import my_njit


@my_njit
def unpack_bits(representation: np.ndarray, n: int = 2):
    """
    For n=2, maps AB => [A, B]
    """

    # extract bits along new dimension (simplifies indexing)
    new_shape = representation.shape + (n,)
    bits = np.empty(new_shape, dtype=np.uint8)

    for i in range(n):
        location = n - i - 1
        bits[..., location] = (representation // (1 << i)) % 2

    return bits


@my_njit
def pack_bits(bits: np.ndarray, dtype=np.uint8):
    """
    Maps [A, B] => AB
    """
    # prepare output
    n = bits.shape[-1]
    representation = np.zeros(bits.shape[:-1], dtype=dtype)

    two_arr = np.array(2, dtype=dtype)
    mul_arr = np.array(1, dtype=dtype)
    for i in range(0, n):
        b = bits[..., (n - i - 1)]
        b = np.asarray(b, dtype=dtype)
        representation += mul_arr * b
        mul_arr *= two_arr

    return representation
