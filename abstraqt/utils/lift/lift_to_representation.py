from typing import Callable

from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.inspection.function_arguments import count_n_arguments, n_arguments_label


def lift_to_representation(clazz: type, f: Callable):
    assert clazz is not None

    @rename_function(f.__name__ + '_representation')
    def ret(*args):
        args = [a.representation for a in args]
        representation = f(*args)
        return clazz(representation)

    try:
        n_arguments = count_n_arguments(f)
        setattr(ret, n_arguments_label, n_arguments)
    except:
        pass

    return ret
