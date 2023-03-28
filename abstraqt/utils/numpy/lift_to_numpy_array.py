import numpy as np

from abstraqt.utils.lift.lift import LiftError


def lift_to_numpy_array(x) -> np.ndarray:
    if isinstance(x, (int, float, complex, bool, np.bool_, np.uint8, np.int8, np.int32, np.int64)):
        ret = np.asarray(x)
    elif isinstance(x, np.ndarray):
        ret = x
    else:
        raise LiftError(x, np.ndarray)
    assert isinstance(ret, np.ndarray)
    return ret
