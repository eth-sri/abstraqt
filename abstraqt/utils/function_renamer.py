def rename_function(new_function_name: str):
    # code inspired by functools.update_wrapper
    def rename(function):
        assigned = ('__name__', '__qualname__')

        for attr in assigned:
            setattr(function, attr, new_function_name)
        return function

    return rename
