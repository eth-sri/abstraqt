from unittest import TestCase

import numpy as np
from cachier import cachier

from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils.cachier_helper import default_cachier_arguments, my_hash
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_abstract_aaronson_gottesman


class TestHash(TestCase):

    def test_hash_class(self):
        h = my_hash(AbstractStabilizerPlus)
        self.assertEqual(h, 7868976617763094669)

    def test_hash(self):
        np.random.seed(0)
        g1 = random_abstract_aaronson_gottesman(2, 2)
        h1 = hash(g1)
        self.assertEqual(h1, 7270078025977617857)

        np.random.seed(0)
        g2 = random_abstract_aaronson_gottesman(2, 2)

        self.assertEqual(g1, g2)
        self.assertEqual(hash(g1), hash(g2))

        d = {g1: 42}
        self.assertEqual(d[g2], 42)

    def test_cached(self):
        g = random_abstract_aaronson_gottesman(2, 2)
        function_to_cache(g)
        cached = function_to_cache(g)

        expected = g.conjugate('X', 0)
        self.assertEqual(cached, expected)


@cachier(**default_cachier_arguments)
def function_to_cache(a: AbstractStabilizerPlus):
    return a.conjugate('X', 0)
