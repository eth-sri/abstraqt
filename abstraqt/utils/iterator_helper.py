from collections.abc import Iterable, Iterator
from typing import Union


def get_unique(it: Union[Iterable, Iterator]):
    """
    :return: The unique concrete element abstracted by self, or a ValueError
    """
    try:
        # convert iterable to iterator
        it = iter(it)
    except TypeError:
        pass  # ignore if we already have an iterator

    try:
        ret = next(it)
    except StopIteration as e:
        raise ValueError(f'No object in {it}') from e

    try:
        other = next(it)
        raise ValueError(f'Multiple objects {ret} and {other} in {it}')
    except StopIteration:
        return ret
