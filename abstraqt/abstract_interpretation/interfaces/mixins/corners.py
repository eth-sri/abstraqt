import itertools
from abc import ABC
from typing import Sequence

import numpy as np


class Corners(ABC):

    def get_corners_single(self):
        """
        Same as get_corners, but assumes shape ()
        """
        raise NotImplementedError()

    @staticmethod
    def empty_reference(shape: Sequence[int], *args, **kwargs):
        """
        Returns an empty instance of the reference implementation
        """
        raise NotImplementedError()

    def get_corners(self, *args, **kwargs):
        """
        Return a Sequence of concrete elements c1, ..., cn:
        - c1 ⊔ ... ⊔ cn = self
        - ci is represented by the reference class
        """

        def get_corners_single(index):
            partial = self[index]
            return partial.get_corners_single(*args, **kwargs)

        per_index_corner_iterators = tuple(
            get_corners_single(index)
            for index in np.ndindex(self.shape)
        )
        combinations = itertools.product(*per_index_corner_iterators)
        for t in combinations:
            t = tuple(t)
            ret = self.empty_reference(self.shape, *args, **kwargs)
            for index, value in zip(np.ndindex(self.shape), t):
                if hasattr(value, 'representation'):
                    value = value.representation
                ret[index] = value

            yield ret

    #####################
    # TO BE IMPLEMENTED #
    #####################

    @property
    def shape(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()
