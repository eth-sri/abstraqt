from abc import ABC
from typing import Sequence


class BottomConstructor(ABC):

    @staticmethod
    def bottom(shape: Sequence[int]):
        raise NotImplementedError()


class DefaultValueConstructors(BottomConstructor, ABC):

    @staticmethod
    def top(shape: Sequence[int]):
        raise NotImplementedError()

    @staticmethod
    def empty(shape: Sequence[int]):
        raise NotImplementedError()
