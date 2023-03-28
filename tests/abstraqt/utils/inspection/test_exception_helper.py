from unittest import TestCase

from abstraqt.utils.inspection.exception_helper import get_exception_information


class ExceptionHelperTest(TestCase):

    def test_get_exception_information(self):
        try:
            assert True
            assert False, 'message'
        except AssertionError:
            filename, line, func, text = get_exception_information()

        self.assertIn('test_exception_helper.py', filename)
        self.assertEqual(line, 11)
        self.assertEqual(func, 'test_get_exception_information')
        # self.assertIn('message', text)
