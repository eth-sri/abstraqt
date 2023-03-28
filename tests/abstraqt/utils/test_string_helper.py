from unittest import TestCase

from abstraqt.utils.string_helper import number_to_human_readable, time_to_human_readable


class IteratorHelperTest(TestCase):

	def test_number_to_human_readable(self):
		self.assertEqual(number_to_human_readable(0.1), '0.1')
		self.assertEqual(number_to_human_readable(11.34), '11')

	def test_time_to_human_readable(self):
		self.assertEqual(time_to_human_readable(0.1), '0.1s')
		self.assertEqual(time_to_human_readable(66), '1.1m')
		self.assertEqual(time_to_human_readable(121), '2.0m')
