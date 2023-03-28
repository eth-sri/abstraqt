from abc import ABC

from abstraqt.abstract_interpretation.interfaces.mixins.default_value_constructors import BottomConstructor
from abstraqt.utils.lift.lift import LiftError


class AbstractLift:

    @classmethod
    def lift(cls, x):
        """
        x: An instance of this class, the reference class, or others
        """
        if isinstance(x, cls):
            return x
        else:
            raise LiftError(x, cls)


class AbstractJoin:

    def join(self, other):
        raise NotImplementedError()


class AbstractLiftJoin(AbstractLift, AbstractJoin, BottomConstructor, ABC):

    @classmethod
    def lift_join(cls, *args, shape=()):
        if len(args) == 0:
            return cls.bottom(shape)

        ret = cls.lift(args[0])
        for arg in args[1:]:
            arg = cls.lift(arg)
            ret = ret.join(arg)
        return ret
