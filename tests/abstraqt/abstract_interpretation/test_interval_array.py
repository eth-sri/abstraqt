import numpy as np

from abstraqt.abstract_interpretation.interval_array import IntervalArray
from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_array import \
    AbstractTestAbstractArrayWrapper


class TestIntervalArray(AbstractTestAbstractArrayWrapper.AbstractTestAbstractArray):

    def __init__(self, *args, **kwargs):
        accept_imprecision = {'cos'}
        super().__init__(IntervalArray, *args, accept_imprecision=accept_imprecision, **kwargs)

    def random_abstract_element(self, shape):
        # random values in [-8, 8], rounded for easier debugging
        representation = np.random.rand(*shape, 2) * 16 - 8
        representation = np.round(representation, decimals=1)

        # handle bottom
        representation[representation[..., 0] > representation[..., 1], :] = np.nan

        return IntervalArray(representation)

    def test_cos(self):
        self.check_operator(IntervalArray.cos, concrete_op=np.cos, n_args=1)

    def test_sum(self):
        self.check_operator(IntervalArray.sum, concrete_op=np.sum, n_args=1)
