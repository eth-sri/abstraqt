from abc import ABC
from typing import Sequence

from .dimension_error import DimensionError


class Broadcasting(ABC):

    def get_broadcast_shape(self, *arrays):
        all_arrays = [self] + [a for a in arrays]
        shapes = [a.shape for a in all_arrays]
        target_shape = broadcast_shapes(*shapes)
        return target_shape

    def broadcast_arrays(self, *arrays):
        """
        Returns read-only views
        """
        target_shape = self.get_broadcast_shape(*arrays)
        all_arrays = [self] + [a for a in arrays]
        return tuple(a.broadcast_to(target_shape) for a in all_arrays)

    def broadcast_to(self, shape: Sequence[int]):
        """
        Broadcast an array to a new shape. Returns a read-only view
        """
        raise NotImplementedError()


def broadcast_shapes(*shapes: Sequence[int]):
    """
    Prefetches numpy implementation from numpy 1.20.0
    """
    n_dimensions = max([len(shape) for shape in shapes])

    ret = []
    for dimension in range(n_dimensions):
        index = -1 - dimension
        sizes = set(get_default(shape, index, 1) for shape in shapes)
        sizes.discard(1)
        if len(sizes) == 0:
            ret.insert(0, 1)
        elif len(sizes) == 1:
            (val,) = sizes
            ret.insert(0, val)
        else:
            raise DimensionError(f'Conflicting shapes in {dimension}-th dimension: {shapes}')

    return ret


def get_default(s: Sequence, index: int, default=None):
    try:
        return s[index]
    except IndexError:
        return default
