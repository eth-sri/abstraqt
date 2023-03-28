from unittest import TestCase

from abstraqt.utils.iterator_helper import get_unique


class IteratorHelperTest(TestCase):

    def test_get_unique(self):
        one = get_unique(range(1))
        self.assertEqual(one, 0)

    def test_get_unique_multiple(self):
        with self.assertRaises(ValueError):
            get_unique(range(2))

    def test_get_unique_none(self):
        with self.assertRaises(ValueError):
            get_unique(range(0))
