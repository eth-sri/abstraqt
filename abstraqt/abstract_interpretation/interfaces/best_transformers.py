import itertools
from typing import Tuple

from cachier import cachier

from abstraqt.utils import logging
from abstraqt.utils.array.lookup_table import LookupTable
from abstraqt.utils.cachier_helper import default_cachier_arguments
from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.inspection.function_arguments import count_n_arguments
from abstraqt.utils.lift.lift_to_representation import lift_to_representation

logger = logging.getLogger(__name__)


def possible_concrete_outputs(concrete_implementation, args, more_args=(), include_inputs=False, **kwargs):
    # prepare relevant concrete inputs
    corner_tuples = itertools.product(*tuple(a.get_corners() for a in args))
    corner_tuples = tuple(corner_tuples)

    # apply reference implementation
    outputs = tuple(concrete_implementation(*corner_tuple, *more_args, **kwargs) for corner_tuple in corner_tuples)

    if include_inputs:
        return zip(corner_tuples, outputs)
    else:
        return outputs


def best_transformer_from_concrete(target_class, concrete_implementation, args: Tuple, target_shape=(), more_args=(),
                                   **kwargs):
    outputs = possible_concrete_outputs(concrete_implementation, args, more_args=more_args, **kwargs)
    if any(o is None for o in outputs):
        result = None
    elif any(hasattr(o, 'encodes_error') and o.encodes_error() for o in outputs):
        # encode error as bottom
        result = target_class.bottom(outputs[0].shape)
    else:
        result = target_class.lift_join(*outputs, shape=target_shape)
    return result


class NoneWrapper:
    pass


@cachier(**default_cachier_arguments)
def best_transformer_from_concrete_cached(target_class, concrete_implementation, args: Tuple, target_shape=(),
                                          more_args=(), **kwargs):
    logger.verbose('Computing (and caching) best transformer for %s', concrete_implementation.__name__)
    logger.slow('Arguments to best transformer for %s: %s %s %s', concrete_implementation.__name__, args, more_args,
                kwargs)
    ret = best_transformer_from_concrete(target_class, concrete_implementation, args,
                                         target_shape=target_shape, more_args=more_args, **kwargs)
    if ret is None:
        return NoneWrapper()
    else:
        return ret


def from_reference_implementation(clazz, reference_implementation, label: str, target_class=None, n_arguments=None,
                                  target_shape=()):
    """
    Parameters
    ----------
    clazz:
        The class for which we are adding an implementation
    reference_implementation :
        A reference implementation operating on instances of the reference class
    label :
        A unique identifier for the resulting implementation
    target_class :
        The output class
    n_arguments :
        Number of expected arguments for reference_implementation (inferred by default)
    """
    if target_class is None:
        target_class = clazz

    @rename_function(reference_implementation.__name__ + '_on_representation')
    def reference_implementation_on_representation(*args):
        # prepare relevant concrete inputs
        args = tuple(clazz(a) for a in args)
        result = best_transformer_from_concrete(target_class, reference_implementation, args, target_shape=target_shape)
        return result.representation

    if n_arguments is None:
        n_arguments = count_n_arguments(reference_implementation)
    bounds = [clazz.bound_on_representation for _ in range(n_arguments)]
    t = LookupTable(reference_implementation_on_representation, bounds, label)
    return t.lookup


def get_best_transformer_on_representation(clazz, function_name: str, reference_implementation, target_class=None,
                                           n_arguments=None, target_shape=()):
    if target_class is None:
        target_class = clazz

    label = clazz.__name__ + '.' + function_name
    lookup = from_reference_implementation(clazz, reference_implementation, label, target_class=target_class,
                                           n_arguments=n_arguments, target_shape=target_shape)

    return lookup


def add_best_transformer_to_class(clazz, function_name: str, reference_implementation, target_class=None,
                                  n_arguments=None, target_shape=()):
    """
    Add a function defined through its reference implementation
    """
    if target_class is None:
        target_class = clazz

    lookup = get_best_transformer_on_representation(
        clazz,
        function_name,
        reference_implementation,
        target_class,
        n_arguments=n_arguments,
        target_shape=target_shape
    )
    setattr(clazz, function_name, lift_to_representation(target_class, lookup))
    logger.verbose(f'Added %s to %s', function_name, clazz.__name__)
    return lookup
