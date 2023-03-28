import itertools

import numpy as np

from abstraqt.linalg.common_matrices import matrices_all_dict, get_n_qubits
from abstraqt.linalg.reorder_dimensions import pad_matrix_by_identities
from abstraqt.pauli.pauli_reference import conjugate_with_matrix
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.iterator_helper import get_unique
from tests.abstraqt.stabilizer.check_operator_via_concretization import WrapperCheckOperator
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans


class TestMoveStabilizerPrefactors(WrapperCheckOperator.CheckOperator):

    def test_move(self):
        for seed, g in random_aaronson_gottesmans(point=True):
            g_after = g.move_stabilizer_prefactors()

            density_expected = get_unique(g.get_densities())
            density_actual = get_unique(g_after.get_densities())

            np.testing.assert_allclose(density_expected, density_actual)

    def test_move_abstract(self):
        self.check_operator_via_best_transformer_wrapper(
            AbstractStabilizerPlus.move_stabilizer_prefactors,
        )
