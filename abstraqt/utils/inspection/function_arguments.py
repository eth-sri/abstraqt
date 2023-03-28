from inspect import signature

n_arguments_label = '__n_arguments__'


def count_n_arguments(f):
    if hasattr(f, n_arguments_label):
        return getattr(f, n_arguments_label)

    sig = signature(f)

    n_arguments = 0

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL:
            raise ValueError(f'Cannot count number of *args for {f}')

        if param.default is param.empty:
            n_arguments += 1

    return n_arguments
