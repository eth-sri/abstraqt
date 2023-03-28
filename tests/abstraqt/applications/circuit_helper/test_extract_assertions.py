from unittest import TestCase

from abstraqt.applications.circuit_helper.extract_assertions import extract_assertions_from_lines


class TestExtractAssertions(TestCase):

    def test_extract_assertions(self):
        assertions = extract_assertions_from_lines(['', '// assert bases |00> + |++>'])
        self.assertEqual(['00', '++'], assertions)
