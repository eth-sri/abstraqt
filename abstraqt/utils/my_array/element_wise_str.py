import sys

import numpy as np

from abstraqt.utils.string_helper import default_element_wise_str_threshold


class ElementWiseStr:
    element_wise_str_threshold = default_element_wise_str_threshold

    def __str__(self):
        ret = self.repr_element_wise()
        with np.printoptions(threshold=self.element_wise_str_threshold):
            return str(ret)

    def __repr__(self):
        ret = self.repr_element_wise()
        with np.printoptions(threshold=sys.maxsize):
            return str(ret)

    def repr_element_wise(self):
        if hasattr(self, 'shape'):
            shape = self.shape
        else:
            shape = self.representation.shape
        ret = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            ret[index] = self.repr_single(self.representation[index])
        return ret

    @staticmethod
    def repr_single(v):
        """
        Same as __repr__, but takes a single element as input
        """
        raise NotImplementedError()
