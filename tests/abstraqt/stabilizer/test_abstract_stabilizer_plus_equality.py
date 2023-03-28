from unittest import TestCase

from abstraqt.abstract_interpretation import IntervalArray
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans


class CheckEqualityWrapper:
    class CheckEquality(TestCase):

        def check_is_super_set(self, a: AbstractStabilizerPlus, b: AbstractStabilizerPlus):
            self.assertEqual(a.n_bits, b.n_bits)

            for basis in ['Z', '-Z', 'X', '-X']:
                for i in range(a.n_bits):
                    prob_a = a.measure(basis, i).get_trace()
                    prob_b = b.measure(basis, i).get_trace()

                    self.assertIsInstance(prob_a, IntervalArray)
                    self.assertIsInstance(prob_b, IntervalArray)

                    msg = f'Violated {a} ⊇ {b}: Measuring bit {i} in basis {basis} yields probability {prob_a} which is not a superset of {prob_b}'
                    self.assertTrue(prob_a.is_super_set_of(prob_b), msg)


class TestEquality(CheckEqualityWrapper.CheckEquality):

    def test_is_super_set(self):
        for seed, g in random_aaronson_gottesmans(point=False):
            with self.subTest(seed=seed, g=g):
                self.check_is_super_set(g, g)
