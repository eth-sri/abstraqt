import itertools

import numpy as np

from abstraqt.linalg.common_matrices import matrices_all_dict, get_n_qubits
from abstraqt.linalg.reorder_dimensions import pad_matrix_by_identities
from abstraqt.pauli.pauli_reference import conjugate_with_matrix
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.iterator_helper import get_unique
from tests.abstraqt.stabilizer.check_operator_via_concretization import WrapperCheckOperator
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans


class TestConjugate(WrapperCheckOperator.CheckOperator):

    #############
    # CONJUGATE #
    #############

    def test_conjugate(self):
        for seed, g in random_aaronson_gottesmans(point=True):
            self.check_conjugate(g, seed=seed)

    def test_conjugate_specific(self):
        g = AbstractStabilizerPlus.zero_state(3)
        g = self.check_conjugate_specific(g, 'H', 0)
        g = self.check_conjugate_specific(g, 'H', 1)
        g = self.check_conjugate_specific(g, 'X', 2)
        g = self.check_conjugate_specific(g, 'CNOT', 1, 2)
        g = self.check_conjugate_specific(g, 'CNOT', 0, 2)
        g = self.check_conjugate_specific(g, 'H', 0)
        g = self.check_conjugate_specific(g, 'H', 1)
        g = self.check_conjugate_specific(g, 'H', 2)

    def check_conjugate(self, g: AbstractStabilizerPlus, **description):
        for m, *positions in conjugate_arguments(g):
            self.check_conjugate_specific(g, m, *positions, **description)

    def check_conjugate_specific(self, g: AbstractStabilizerPlus, m, *positions, **description):
        with self.subTest(n_bits=g.n_bits, n_summands=g.n_summands, m=m, positions=positions, **description):
            density = get_unique(g.get_densities())
            m_positioned = pad_matrix_by_identities(m, g.n_bits, *positions)
            density_expected = conjugate_with_matrix(m_positioned)(density)

            g_after = g.conjugate(m, *positions)
            density_actual = get_unique(g_after.get_densities())

            np.testing.assert_allclose(density_actual, density_expected, atol=1e-7)

            return g_after

    def test_conjugate_decomposed_abstract(self):
        self.check_operator_via_best_transformer_wrapper(
            AbstractStabilizerPlus.conjugate_decomposed,
            conjugate_arguments
        )

    def test_conjugate_pauli_abstract(self):
        self.check_operator_via_best_transformer_wrapper(
            AbstractStabilizerPlus.conjugate_paulis,
            conjugate_arguments
        )


def conjugate_arguments(g: AbstractStabilizerPlus):
    n_bits = g.n_bits
    for m in matrices_all_dict.keys():
        for positions in itertools.combinations(list(range(n_bits)), get_n_qubits(m)):
            yield m, *positions
