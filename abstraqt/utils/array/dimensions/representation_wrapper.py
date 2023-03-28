from typing import Sequence

import numpy as np

from .broadcasting import Broadcasting
from .shaping import Shaping
from ...hash_helper import my_numpy_hash


class RepresentationItem:

    def __getitem__(self, item):
        ret = self.representation[item]
        ret = self.__class__(ret)
        return ret

    def __setitem__(self, key, value):
        if hasattr(value, 'representation'):
            value = value.representation
        self.representation[key] = value


class RepresentationShape:

    @property
    def shape(self):
        return self.representation.shape


class RepresentationBroadcast(Broadcasting):

    def broadcast_to(self, shape: Sequence[int]):
        representation = np.broadcast_to(self.representation, shape)
        return self.__class__(representation)


class RepresentationShaping(Shaping):

    def reshape(self, new_shape: Sequence[int]):
        representation = self.representation.reshape(new_shape)
        return self.__class__(representation)

    def swapaxes(self, axis1: int, axis2: int):
        representation = np.swapaxes(self.representation, axis1, axis2)
        return self.__class__(representation)


class RepresentationRepeat:

    def repeat(self, repeats, axis=None):
        """
        https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
        """
        ret = np.repeat(self.representation, repeats, axis=axis)
        return self.__class__(ret)


class RepresentationCopy:

    def copy(self):
        return self.__class__(self.representation.copy())


class RepresentationStr:

    def __str__(self):
        return str(repr(self))

    def __repr__(self):
        return f'({self.__class__.__name__}) {self.representation}'


class RepresentationHash:

    def stable_representation_hash(self):
        return my_numpy_hash(self.representation)


class RepresentationWrapper(
    RepresentationHash,
    RepresentationShape,
    RepresentationItem,
    RepresentationBroadcast,
    RepresentationShaping,
    RepresentationStr,
    RepresentationCopy,
    RepresentationRepeat,
):
    pass
