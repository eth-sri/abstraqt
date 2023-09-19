from abstraqt.abstract_interpretation.abstract_int2_array import AbstractInt2Array
from abstraqt.utils.my_array.mod_array import Mod4Array
from tests.abstraqt.abstract_interpretation.interfaces.abstract_test_abstract_bit_pattern_array import \
    AbstractTestAbstractBitPatternArrayWrapper


class TestAbstractInt2Array(AbstractTestAbstractBitPatternArrayWrapper.AbstractTestAbstractBitPatternArray):

    def __init__(self, *args, **kwargs):
        super().__init__(AbstractInt2Array, *args, **kwargs)

    def test_sum(self):
        self.check_operator(AbstractInt2Array.sum, concrete_op=Mod4Array.sum)

    def test_sum_axis(self):
        self.check_operator(AbstractInt2Array.sum, concrete_op=Mod4Array.sum, axis=0)

    def test_dot(self):
        self.check_operator(AbstractInt2Array.dot, concrete_op=Mod4Array.dot)

    def test_sum_n_times(self):
        top = AbstractInt2Array.top(())
        s = top.sum_n_times(0)
        zero = AbstractInt2Array.lift(0)
        self.assertTrue(s.equal_abstract_object(zero))

        s = top.sum_n_times(1)
        self.assertTrue(s.equal_abstract_object(top))

        s = top.sum_n_times(2)
        self.assertTrue(s.equal_abstract_object(top))

        one = AbstractInt2Array.lift(1)
        zero_one = zero.join(one)
        s = zero_one.sum_n_times(2)
        zero_one_two = zero_one.join(AbstractInt2Array.lift(2))
        self.assertTrue(s.equal_abstract_object(zero_one_two))
