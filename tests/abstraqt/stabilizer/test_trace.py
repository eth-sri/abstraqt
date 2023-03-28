import numpy as np

from abstraqt.abstract_interpretation import IntervalArray
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.iterator_helper import get_unique
from tests.abstraqt.stabilizer.check_operator_via_concretization import WrapperCheckOperator
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans


class TestTrace(WrapperCheckOperator.CheckOperator):

    def test_trace(self):
        for seed, g in random_aaronson_gottesmans(point=True):
            with self.subTest(n_bits=g.n_bits, n_summands=g.n_summands, seed=seed, g=g):
                actual_trace = g.get_trace()
                actual_trace = get_unique(actual_trace.get_corners())

                density = get_unique(g.get_densities())
                expected_trace = np.trace(density).real

                np.testing.assert_almost_equal(actual_trace, expected_trace)

    def test_trace_abstract(self):
        for seed, g in random_aaronson_gottesmans(point=False):
            with self.subTest(seed=seed, g=g):
                self.check_operator_via_best_transformer(
                    AbstractStabilizerPlus.get_trace,
                    g,
                    IntervalArray,
                    (g.n_summands, g.n_bits)
                )
