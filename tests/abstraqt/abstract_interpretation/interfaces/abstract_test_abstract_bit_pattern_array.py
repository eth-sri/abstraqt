import numpy as np

from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_array import \
    AbstractTestAbstractArrayWrapper


def random_abstract_bit_pattern_array(clazz, shape, non_bottom=False):
    bound = clazz.bound_on_representation
    start = 1 if non_bottom else 0
    representation = np.random.randint(start, bound, size=shape, dtype=np.uint8)
    ret = clazz(representation)
    return ret


class AbstractTestAbstractBitPatternArrayWrapper:
    """
    Wrapper is needed to avoid running the abstract tests directly
    """

    class AbstractTestAbstractBitPatternArray(AbstractTestAbstractArrayWrapper.AbstractTestAbstractArray):

        def __init__(
                self,
                class_under_test,
                *args,
                **kwargs
        ):
            super().__init__(class_under_test, *args, **kwargs)

        def random_abstract_element(self, shape):
            ret = random_abstract_bit_pattern_array(self.class_under_test, shape)
            return ret
