import numpy as np

from abstraqt.abstract_interpretation import IntervalArray
from abstraqt.linalg.reorder_dimensions import pad_matrix_by_identities
from abstraqt.pauli.pauli_reference import measure_with_matrix
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils import logging
from abstraqt.utils.iterator_helper import get_unique
from tests.abstraqt.abstract_interpretation.interfaces.equal_abstract_object import EqualAbstractObject
from tests.abstraqt.stabilizer.check_operator_via_concretization import WrapperCheckOperator
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans

logger = logging.getLogger(__name__)


class TestMeasure(WrapperCheckOperator.CheckOperator, EqualAbstractObject):

    def test_measure(self):
        for seed, g in random_aaronson_gottesmans(point=True):
            for basis, qubit in measurement_arguments(g):
                with self.subTest(n_bits=g.n_bits, n_summands=g.n_summands, seed=seed, g=g, basis=basis):
                    self.check_measure(g, basis, qubit)

    def test_measure_multiple(self):
        g = AbstractStabilizerPlus.zero_state(2)

        # measure 00
        g_measured = g.measure('ZZ', 0, 1)
        t = g_measured.get_trace()
        self.assert_equal_abstract_object(t, IntervalArray.lift(1))

        # measure ++
        g = g.conjugate('H', 0)
        g = g.conjugate('H', 1)
        g_measured = g.measure('ZZ', 0, 1)
        t = g_measured.get_trace()
        self.assertIsInstance(t, IntervalArray)
        self.assert_equal_abstract_object(t, IntervalArray.lift(1/2))

    @staticmethod
    def check_measure(g: AbstractStabilizerPlus, basis: str, qubit: int):
        density = get_unique(g.get_densities())
        m_positioned = pad_matrix_by_identities(basis, g.n_bits, qubit)
        density_expected = measure_with_matrix(m_positioned, density)

        g_after = g.measure(basis, qubit)
        density_actual = get_unique(g_after.get_densities())

        np.testing.assert_allclose(density_actual, density_expected, atol=1e-7)

    def test_measure_abstract(self):
        for seed, g in random_aaronson_gottesmans(point=False):
            for basis, qubit in measurement_arguments(g):
                with self.subTest(seed=seed, basis=basis, qubit=qubit, g=g):
                    self.check_operator_via_best_transformer(
                        AbstractStabilizerPlus.measure,
                        g,
                        AbstractStabilizerPlus,
                        (g.n_summands, g.n_bits),
                        more_args=(basis, qubit)
                    )


def measurement_arguments(g: AbstractStabilizerPlus):
    for basis in ['X', 'Y', 'Z', '-Z']:
        for qubit in range(g.n_bits):
            yield basis, qubit
