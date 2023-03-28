from typing import Sequence

import numpy as np

from abstraqt.abstract_interpretation import AbstractInt2Array
from abstraqt.abstract_interpretation.abstract_complex_array import AbstractComplexArray
from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_array import \
    AbstractTestAbstractArrayWrapper


def random_abstract_complex_array(shape: Sequence[int], non_bottom=False):
    # random values in [-2, 2], rounded for easier debugging
    representation = np.random.rand(*shape, 2, 2) * 4 - 2
    # adapt phi to cover full range
    representation[..., 1, :] *= 2 * np.pi
    representation = np.round(representation, decimals=1)

    # handle bottom
    bottoms = representation[..., 0] > representation[..., 1]
    if non_bottom:
        representation[bottoms, 1] = representation[bottoms, 0]
    else:
        representation[bottoms, :] = np.nan

    return AbstractComplexArray(representation)


class TestAbstractComplexArray(AbstractTestAbstractArrayWrapper.AbstractTestAbstractArray):

    def __init__(self, *args, **kwargs):
        accept_imprecision = {
            'mul',  # mul does not have a unique best transformer because join is not unique
            'conjugate'  # conjugate does not have a unique best transformer because join is not unique
        }
        super().__init__(
            AbstractComplexArray,
            *args,
            accept_imprecision=accept_imprecision,
            **kwargs
        )

    def random_abstract_element(self, shape):
        ret = random_abstract_complex_array(shape)
        return ret

    def test_real(self):
        self.check_operator(AbstractComplexArray.real, concrete_op=np.real, n_args=1)

    def test_conjugate(self):
        self.check_operator(AbstractComplexArray.conjugate, concrete_op=np.conjugate, n_args=1)

    def test_exponent_base_i(self):
        for n in [1, 2]:
            for e in range(4):
                with self.subTest(n=n, e=e):
                    exponent = np.array(n * [e], dtype=int)
                    a = AbstractInt2Array.lift(exponent)
                    c = a.exponent_base_i()
                    self.assertEqual(exponent.shape, c.shape)

                    expected = AbstractComplexArray.lift(np.power(1j, exponent))

                    self.assert_equal_abstract_object(c, expected)
