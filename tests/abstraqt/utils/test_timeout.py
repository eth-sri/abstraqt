from unittest import TestCase

from abstraqt.utils.timeout import timeout_after, TimedOutException


class TestAbortAfter(TestCase):

    def test_timeout_after_normal(self):
        with timeout_after(1):
            pass

    def test_timeout_after_abort(self):
        with self.assertRaises(TimedOutException):
            with timeout_after(1):
                # infinite loop
                x = 0
                while x % 2 == 0:
                    x += 2
