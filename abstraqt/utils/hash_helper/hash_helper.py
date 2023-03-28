import hashlib
import inspect
from typing import Union

import numpy as np


def my_hashlib_hash(a: Union[str, bytes]):
    # Avoid Python's default hash, which is randomized across different runs
    return int(hashlib.sha512(a).hexdigest()[:16], 16) - 2 ** 63


def my_string_hash(a: str):
    return my_hashlib_hash(a.encode('utf-8'))


def my_bytes_hash(a: bytes):
    return my_hashlib_hash(a)


def _my_numpy_hash(a: np.ndarray, decimals=None):
    if decimals:
        a = a * (10 ** decimals)
        a = np.around(a).astype(int)

    b = a.data.tobytes()
    return my_bytes_hash(b)


def my_numpy_hash(a: np.ndarray, decimals=None):
    if np.iscomplexobj(a):
        if decimals is None:
            raise ValueError('Hashing complex numbers requires rounding to avoid non-deterministic results')
        r = _my_numpy_hash(np.real(a), decimals=decimals)
        i = _my_numpy_hash(np.imag(a), decimals=decimals)
        ret = hash((r, i))
    else:
        ret = _my_numpy_hash(a)
    return ret


def my_hash(a):
    if isinstance(a, str):
        ret = my_string_hash(a)
    elif inspect.isclass(a):
        ret = my_string_hash(a.__name__)
    elif callable(a):
        ret = my_string_hash(a.__name__)
    elif isinstance(a, dict):
        ret = my_hash(tuple((my_hash(k), my_hash(v)) for k, v in a.items()))
    elif isinstance(a, tuple):
        ret = hash(tuple(my_hash(e) for e in a))
    elif isinstance(a, np.ndarray):
        ret = my_numpy_hash(a)
    else:
        ret = hash(a)
    return ret
