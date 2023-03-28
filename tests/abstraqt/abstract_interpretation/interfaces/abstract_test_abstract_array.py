from typing import Set
from unittest import TestCase

import numpy as np

from abstraqt.abstract_interpretation.interfaces.best_transformers import best_transformer_from_concrete, \
    possible_concrete_outputs
from abstraqt.utils.array.equality import filter_equal
from abstraqt.utils.inspection.function_arguments import count_n_arguments
from abstraqt.utils.inspection.operator_helper import get_operators
from abstraqt.utils.iterator_helper import get_unique
from tests.abstraqt.abstract_interpretation.interfaces.equal_abstract_object import EqualAbstractObject
from tests.test_config import default_random_repetitions, default_shapes_list


class AbstractTestAbstractArrayWrapper:
    """
    Wrapper is needed to avoid running the abstract tests directly
    """

    class AbstractTestAbstractArray(EqualAbstractObject, TestCase):

        def __init__(
                self,
                class_under_test,
                *args,
                ignore_operations: Set[str] = None,
                accept_imprecision: Set[str] = None,
                **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.class_under_test = class_under_test
            if ignore_operations is None:
                self.ignore_operations = set()
            else:
                self.ignore_operations = ignore_operations
            if accept_imprecision is None:
                self.accept_imprecision = set()
            else:
                self.accept_imprecision = accept_imprecision

            # always ignore some operations
            for op in ['__getitem__', '__setitem__', '__contains__', '__eq__', '__ne__']:
                self.ignore_operations.add(op)

        ###########
        # CORNERS #
        ###########

        def test_corners_bottom(self):
            bottom = self.class_under_test.bottom(())
            self.assertEqual(list(bottom.get_corners_single()), [])
            self.assertEqual(list(bottom.get_corners()), [])

            bottom2 = self.class_under_test.bottom((2,))
            self.assertEqual(list(bottom2.get_corners()), [])

        def test_corners_top_0D(self):
            top = self.class_under_test.top(())
            corners1 = list(top.get_corners_single())
            corners2 = list(top.get_corners())
            self.assertEqual(corners1, corners2)

        def test_corners_top(self):
            top = self.class_under_test.top((2,))
            for corner in top.get_corners():
                self.assertEqual(corner.shape, (2,))

        def test_lift_corners(self):
            for n in [1, 2]:
                top = self.class_under_test.top((n,))
                for one_concrete in top.get_corners():
                    with self.subTest(n=n, c=one_concrete):
                        abstracted = self.class_under_test.lift(one_concrete)
                        # ensure that we can get the concretization back
                        _ = get_unique(abstracted.get_corners())
                    # do not check for equality as equality may not be implemented

        ########
        # SIZE #
        ########

        def test_size(self):
            for shape in [(), (2,), (2, 2)]:
                with self.subTest(shape):
                    bottom = self.class_under_test.bottom(shape)
                    s = bottom.size()
                    self.assertEqual(s, 0)

                    top = self.class_under_test.top(shape)
                    s = top.size()
                    self.assertEqual(s, 1)

        ##############
        # COMPARISON #
        ##############

        def test_equal_abstract_object(self):
            for shape in default_shapes_list:
                for seed in range(default_random_repetitions):
                    a = self.random_abstract_element(shape)
                    with self.subTest(a, shape=shape, seed=seed):
                        self.assert_equal_abstract_object(a, a)

        def test_contains_reference(self):
            for shape in default_shapes_list:
                for seed in range(default_random_repetitions):
                    np.random.seed(seed)
                    a = self.random_abstract_element(shape)
                    for corner in a.get_corners():
                        with self.subTest(a, corner=corner, shape=shape, seed=seed):
                            lifted = self.class_under_test.lift(corner)
                            msg = f'\n{a}\n does not contain \n{corner},\nwhich abstracts to \n{lifted}'
                            contains = a.contains_reference(corner)
                            self.assertTrue(contains, msg=msg)

        #############
        # OPERATORS #
        #############

        def random_abstract_element(self, shape):
            raise NotImplementedError()

        def test_operators(self):
            for op in self.get_operators():
                self.check_operator(op)

        def check_operator(self, abstract_op, more_args=(), concrete_op=None, n_args=None, **kwargs):
            if n_args is None:
                # infer number of parameters
                n_args = count_n_arguments(abstract_op)

            # determine concrete operation
            if concrete_op is None:
                concrete_op = abstract_op

            for shape in default_shapes_list:
                for seed in range(default_random_repetitions):
                    self.check_operator_with_args(abstract_op, concrete_op, n_args, shape, seed, more_args=more_args,
                                                  **kwargs)

        def check_operator_with_args(self, abstract_op, concrete_op, n_args, shape, seed, more_args=(), **kwargs):
            np.random.seed(seed)
            args = tuple(self.random_abstract_element(shape) for _ in range(n_args))
            with self.subTest(op=abstract_op.__name__, shape=shape, seed=seed, args=args):
                abstract_result = abstract_op(*args, *more_args, **kwargs)
                self.assertIsNotNone(abstract_result)

                # check soundness with concrete outputs
                concrete_outputs = list(possible_concrete_outputs(concrete_op, args, more_args=more_args, **kwargs))
                if any(hasattr(o, 'encodes_error') and o.encodes_error() for o in concrete_outputs):
                    self.assertTrue(abstract_result.is_bottom())
                else:
                    for reference in concrete_outputs:
                        msg = f'Unsound, as expected value\n{reference}\nis not in\n{abstract_result}'
                        contains = abstract_result.contains_reference(reference)
                        self.assertTrue(contains, msg=msg)

                # check best transformer
                abstract_op_name = abstract_op.__name__
                if abstract_op_name not in self.accept_imprecision:
                    expected_result = best_transformer_from_concrete(abstract_result.__class__, concrete_op, args,
                                                                     more_args=more_args, **kwargs)
                    self.assert_equal_abstract_object(abstract_result, expected_result)

        def get_operators(self):
            for name, op in get_operators().items():
                if hasattr(self.class_under_test, name):
                    if name not in self.ignore_operations:
                        n_args = count_n_arguments(op)

                        # check if supported
                        with self.subTest(type="supported?", op=op.__name__):
                            try:
                                args = n_args * (self.class_under_test.top(()),)
                                op(*args)
                            except TypeError as e:
                                if 'not supported between instances' in str(e):
                                    # ignore operations that are not supported
                                    continue
                                else:
                                    raise
                            except NotImplementedError:
                                # skip operations that are not implemented
                                continue
                        yield op

        ###############
        # CONVENIENCE #
        ###############

        def test_canonical(self):
            for shape in default_shapes_list:
                for seed in range(default_random_repetitions):
                    np.random.seed(seed)
                    a = self.random_abstract_element(shape)

                    with self.subTest(shape=shape, seed=seed, a=a):
                        before = filter_equal(a.get_corners())
                        n_before = len(before)

                        a.to_canonical()
                        after = filter_equal(a.get_corners())
                        n_after = len(after)

                        self.assertEqual(n_before, n_after)

        #######
        # STR #
        #######

        def test_str_top(self):
            a = self.class_under_test.top(())
            s = str(a)
            self.assertIsInstance(s, str)
