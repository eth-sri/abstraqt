from operator import __add__, __mul__, __sub__, __matmul__, __neg__, __invert__, __eq__, __and__, __or__, __xor__
from typing import Callable, Sequence

import numpy as np

from abstraqt.utils.array.dimensions.representation_wrapper import RepresentationWrapper
from abstraqt.utils.inspection.function_arguments import count_n_arguments, n_arguments_label
from abstraqt.utils.inspection.operator_helper import get_operator_name


####################
# ADDING OPERATORS #
####################


def add_operator(reference_op, n_arguments=None):
    label = get_operator_name(reference_op)

    def add_operator_to_class(clazz: type):
        wrapped_op = get_mod_operator(reference_op, clazz, n_arguments=n_arguments)
        setattr(clazz, label, wrapped_op)
        return clazz

    return add_operator_to_class


def get_mod_operator(op: Callable, clazz: type, n_arguments=None):
    def mod_operator(*args, **kwargs):
        mod = args[0].mod

        # pre-process input
        args = tuple(a.representation.astype(int) for a in args)

        # perform computation
        ret = op(*args, **kwargs)

        # post-process output
        ret %= mod
        ret = ret.astype(int)

        return clazz(ret)

    if n_arguments is None:
        n_arguments = count_n_arguments(op)
    setattr(mod_operator, n_arguments_label, n_arguments)

    return mod_operator


###########
# CLASSES #
###########


@add_operator(__add__)
@add_operator(__mul__)
@add_operator(__sub__)
@add_operator(__matmul__)
@add_operator(__neg__)
@add_operator(__invert__)
@add_operator(__eq__)
@add_operator(np.dot, 2)
@add_operator(np.sum, 1)
@add_operator(__and__)
@add_operator(__or__)
@add_operator(__xor__)
class ModArray(RepresentationWrapper):

    def __init__(self, representation: np.ndarray):
        self.representation = representation

    mod: int = None

    @classmethod
    def empty(cls, shape: Sequence[int]):
        r = np.empty(shape, dtype=int)
        return cls(r)

    def __int__(self):
        return int(self.representation)


class Mod2Array(ModArray):

    def __init__(self, representation: np.ndarray):
        super().__init__(representation)

    mod = 2

    def all(self, axis=None):
        ret = np.all(self.representation, axis=axis)
        return Mod2Array(ret)


class Mod4Array(ModArray):

    def __init__(self, representation: np.ndarray):
        super().__init__(representation)

    mod = 4
