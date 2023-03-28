import functools
import inspect


def get_class_that_defined_method(meth):
    # Taken from:
    # https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (
            inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__,
                                                                                                '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def get_static_method(clazz, name):
    f = getattr(clazz, name)
    actual_clazz = get_class_that_defined_method(f)
    if actual_clazz == clazz:
        return f
    else:
        return None


def get_object_methods(x):
    object_methods = [
        method_name for method_name in dir(x)
        if callable(getattr(x, method_name))
    ]
    return object_methods
