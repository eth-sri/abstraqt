from unittest import TestCase

from abstraqt.utils.inspection.function_arguments import count_n_arguments


class TestClass:

    def f(self, other):
        return self, other


class TestFunctionArguments(TestCase):

    def test_count_n_arguments(self):
        def f(x, y):
            return x, y

        n_arguments = count_n_arguments(f)
        self.assertEqual(n_arguments, 2)

    def test_count_n_arguments_keyword(self):
        def f(x, y, k=None):
            return x, y, k

        n_arguments = count_n_arguments(f)
        self.assertEqual(n_arguments, 2)

    def test_count_n_arguments_error(self):
        def f(*args):
            return args

        with self.assertRaises(ValueError):
            count_n_arguments(f)

    def test_count_n_arguments_class(self):
        n_arguments = count_n_arguments(TestClass.f)
        self.assertEqual(n_arguments, 2)
