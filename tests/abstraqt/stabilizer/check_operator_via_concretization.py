from unittest import TestCase

from abstraqt.abstract_interpretation.interfaces.best_transformers import best_transformer_from_concrete_cached, \
    NoneWrapper
from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from tests.abstraqt.stabilizer.random_abstract_stabilizer_plus import random_aaronson_gottesmans


class WrapperCheckOperator:
    class CheckOperator(TestCase):

        def check_operator_via_best_transformer(self, op, g: AbstractStabilizerPlus, target_class, target_shape,
                                                more_args=()):
            abstract_result = op(g, *more_args)
            expected_result = best_transformer_from_concrete_cached(
                target_class,
                op,
                (g,),
                target_shape=target_shape,
                more_args=more_args
            )

            if abstract_result is None:
                if not isinstance(expected_result, NoneWrapper):
                    self.assertTrue(expected_result.is_bottom())
                return

            msg = f'Unsound, as best transformer yields {expected_result}\nwhich is not a subset of\n{abstract_result}'
            contains = abstract_result.is_super_set_of(expected_result)
            self.assertTrue(contains, msg=msg)

        def check_operator_via_best_transformer_wrapper(self, op, get_more_arguments=lambda _: ()):
            for seed, g in random_aaronson_gottesmans(point=False, label=op.__name__):
                for more_args in get_more_arguments(g):
                    with self.subTest(seed=seed, n_bits=g.n_bits, n_summands=g.n_summands, args=more_args,
                                      g=g):
                        self.check_operator_via_best_transformer(
                            op,
                            g,
                            AbstractStabilizerPlus,
                            (g.n_summands, g.n_bits),
                            more_args=more_args
                        )
