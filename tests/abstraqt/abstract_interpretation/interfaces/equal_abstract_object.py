class EqualAbstractObject:

    def assert_equal_abstract_object(self, actual, expected):
        actual.to_canonical()
        expected.to_canonical()
        self.assertTrue(actual.equal_abstract_object(expected), f'Expected {expected} but got {actual}')
