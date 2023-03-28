import numpy as np

from abstraqt.abstract_interpretation.abstract_bool_array import AbstractBoolArray, solve_mod_2_abstract_rhs
from abstraqt.abstract_interpretation.interfaces.best_transformers import best_transformer_from_concrete
from abstraqt.linalg.solve_mod_2 import solve_mod_2
from abstraqt.utils.array.mod_array import Mod2Array
from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_bit_pattern_array import \
    AbstractTestAbstractBitPatternArrayWrapper
from tests.abstraqt.linalg.random_matrix import get_bool_matrix
from tests.test_config import default_random_repetitions


class TestAbstractBoolArray(AbstractTestAbstractBitPatternArrayWrapper.AbstractTestAbstractBitPatternArray):

    def __init__(self, *args, **kwargs):
        super().__init__(AbstractBoolArray, *args, **kwargs)

    def test_sum(self):
        self.check_operator(AbstractBoolArray.sum, concrete_op=Mod2Array.sum)

    def test_sum_axis(self):
        self.check_operator(AbstractBoolArray.sum, concrete_op=Mod2Array.sum, axis=0)

    def test_dot(self):
        self.check_operator(AbstractBoolArray.dot, concrete_op=Mod2Array.dot)

    def test_all(self):
        self.check_operator(AbstractBoolArray.all, concrete_op=Mod2Array.all)

    def test_solve_abstract(self):
        for a_shape in [(1, 1), (2, 2), (3, 3)]:
            for seed in range(default_random_repetitions):
                np.random.seed(seed)
                a = get_bool_matrix(*a_shape, full_rank=True)
                b = self.random_abstract_element(a_shape[0])
                with self.subTest(shape=a_shape, a=a, b=b, seed=seed):
                    abstract_result = solve_mod_2_abstract_rhs(a, b)
                    expected_result = best_transformer_from_concrete(AbstractBoolArray, solve_mod_2_wrapper, (b,), a=a)
                    self.assert_equal_abstract_object(abstract_result, expected_result)


def solve_mod_2_wrapper(b: Mod2Array, a: np.ndarray):
    b = b.representation
    x = solve_mod_2(a, b)
    x = Mod2Array(x)
    return x
