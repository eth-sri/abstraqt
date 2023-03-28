from unittest import TestCase

from abstraqt.utils.function_renamer import rename_function


class TestFunctionRenamer(TestCase):

    def test_rename_function(self):
        @rename_function('very_specific_string')
        def f(x):
            return x

        self.assertEqual(f.__name__, 'very_specific_string')

        s = str(f)
        self.assertIn('very_specific_string', s)

        r = repr(f)
        self.assertIn('very_specific_string', r)
