import numpy as np

from abstraqt.utils.hash_helper.hash_helper import my_numpy_hash


class HashableNumpyWrapper:

    def __init__(self, x: np.ndarray, decimals=3):
        assert isinstance(x, np.ndarray)
        self.x = x
        self.decimals = decimals

    def __hash__(self):
        return my_numpy_hash(self.x, decimals=self.decimals)

    def __eq__(self, other):
        if not isinstance(other, HashableNumpyWrapper):
            return False
        return np.allclose(self.x, other.x)
