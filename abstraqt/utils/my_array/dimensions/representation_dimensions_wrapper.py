from typing import Sequence

import numpy as np


class RepresentationDimensionsWrapper:

    def __init__(self, n_dimensions: int):
        assert isinstance(n_dimensions, int)
        self.n_dimensions = n_dimensions

    @property
    def shape(self):
        return self.representation.shape[:-self.n_dimensions]

    def __getitem__(self, item):
        item = add_last(item, self.n_dimensions)
        ret = self.representation[item]
        ret = self.__class__(ret)
        return ret

    def __setitem__(self, key, value):
        key = add_last(key, self.n_dimensions)

        if hasattr(value, 'representation'):
            value = value.representation
        self.representation[key] = value

    def reshape(self, new_shape: Sequence[int]):
        overall_new_shape = tuple(new_shape) + self.representation.shape[-self.n_dimensions:]
        representation = self.representation.reshape(overall_new_shape)
        return self.__class__(representation)

    def swapaxes(self, axis1: int, axis2: int):
        axis1 = adapt_axis(axis1, self.n_dimensions)
        axis2 = adapt_axis(axis2, self.n_dimensions)

        representation = np.swapaxes(self.representation, axis1, axis2)
        return self.__class__(representation)

    def copy(self):
        return self.__class__(self.representation.copy())


def add_last(index, n):
    if not isinstance(index, tuple):
        index = (index,)
    index = index + n * (slice(None),)
    return index


def adapt_axis(axis: int, n_skipped_dimensions: int):
    if axis < 0:
        return axis - n_skipped_dimensions
    else:
        return axis
