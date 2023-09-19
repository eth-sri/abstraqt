from abc import ABC
from typing import Sequence, Union

import numpy as np


class Shaping(ABC):

    #############
    # RESHAPING #
    #############

    def reshape(self, new_shape: Sequence[int]):
        """
        Gives a new shape to an array without changing its data.
        Returns a view.
        """
        raise NotImplementedError()

    def expand_dims(self, axis: Union[int, Sequence[int]]):
        """
        Insert new axes that will appear at the axis position
        in the expanded array shape. Returns a view.
        """
        new_shape = [n for n in self.shape]
        if isinstance(axis, int):
            axis = [axis]

        pos_axis = [a for a in axis if a >= 0]
        pos_axis.sort()
        for a in pos_axis:
            new_shape.insert(a, 1)

        neg_axis = [a for a in axis if a < 0]
        neg_axis.sort(reverse=True)
        for a in neg_axis:
            # warning: index -1 does not correspond to the last entry
            new_shape.insert(len(new_shape) + 1 + a, 1)

        return self.reshape(new_shape)

    def flatten(self):
        """
        Collapse into one dimension
        """
        n = np.prod(self.shape)
        n = int(n)
        ret = self.reshape((n,))
        return ret


def pairing(xs, ys, ignore_last_n_dimensions=0):
    """
    Reshape xs and ys such that applying any function f on them yields the result for all pairs f(x,y) for
    - x from xs and
    - y from ys

    Example:
    - xs.shape = [a,b,c]
    - ys.shape = [d,e,c]
    - ignore_last_n_dimensions = 1
    =>
    - xs.shape = [a,b,1,1,c]
    - ys.shape = [1,1,d,e,c]
    """
    if ignore_last_n_dimensions > 0:
        end_x = xs.shape[-ignore_last_n_dimensions:]
        end_y = ys.shape[-ignore_last_n_dimensions:]
        assert end_x == end_y, f'Mismatch between {end_x} and {end_y}'

    n_dim_xs = len(xs.shape) - ignore_last_n_dimensions
    n_dim_ys = len(ys.shape) - ignore_last_n_dimensions

    r1_extend = [n_dim_xs + m for m in range(n_dim_ys)]
    r2_extend = [m for m in range(n_dim_xs)]

    r1 = expand_dims(xs, r1_extend)
    r2 = expand_dims(ys, r2_extend)

    return r1, r2


def expand_dims(x, *args):
    if hasattr(x, 'expand_dims'):
        return x.expand_dims(*args)
    else:
        return np.expand_dims(x, *args)
