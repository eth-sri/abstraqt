import inspect
import operator


def _get_operator_name(op):
    label = op.__name__

    if not label.startswith('__'):
        label = '__' + label
    if not label.endswith('__'):
        if label.endswith('_'):
            label += '_'
        else:
            label += '__'
    return label


def get_operators():
    ret = {}
    for _, member in inspect.getmembers(operator):
        if member.__class__.__name__ == 'builtin_function_or_method':
            name = _get_operator_name(member)
            ret[name] = member
    return ret


_operators = get_operators()


def get_operator_name(op):
    if op in _operators.values():
        return _get_operator_name(op)
    else:
        return op.__name__
