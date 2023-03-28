import numpy as np

from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans
from tests.abstraqt.stabilizer.test_abstract_stabilizer_plus_equality import CheckEqualityWrapper


class TestCompression(CheckEqualityWrapper.CheckEquality):

    def test_repeated_conjugate_decomposed(self):
        n_expansions = 14
        for start_with_h in [False, True]:
            g = AbstractStabilizerPlus.zero_state(1)

            if start_with_h:
                g = g.conjugate('H', 0)

            for i in range(n_expansions):
                # expand shape
                g = g.conjugate_decomposed('I', 0)

                # run operations on expanded shape
                g_conjugated = g.conjugate('I', 0)
                g_measured = g.measure('Z', 0)
                g.get_trace()

                expected_number_of_dimensions = 2 * (i + 1) + 1
                for g_check in [g, g_conjugated, g_measured]:
                    np.testing.assert_equal(g_check.position_in_sum, np.zeros((1, expected_number_of_dimensions)))

    def test_position_in_sum(self):
        g = AbstractStabilizerPlus.zero_state(1)
        np.testing.assert_equal(g.position_in_sum, np.array([
            [0]
        ]))

        g = g.conjugate_decomposed('T', 0)
        np.testing.assert_equal(g.position_in_sum, np.array([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1]
        ]))

        g = g.conjugate_decomposed('T', 0)
        np.testing.assert_equal(g.position_in_sum, np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],

            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1],

            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],

            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1]
        ]))

    def test_repeated_compression(self):
        n_expansions = 14

        g = AbstractStabilizerPlus.zero_state(1)

        for i in range(n_expansions):
            # expand shape
            g = g.conjugate_decomposed('T', 0)

            g = g.compress(1)

            self.assertLessEqual(g.n_summands, 1)

    def test_compression(self):
        for seed, g in random_aaronson_gottesmans(point=False):
            with self.subTest(seed=seed, g=g):
                g = g.conjugate_decomposed('T', 0)
                g_compressed = g.compress(1)

                self.check_is_super_set(g_compressed, g)
