import time
from typing import Callable

from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.logging.log_data import report_runtime_decorator


def log_calls(logger):
    def wrap(f: Callable):
        @report_runtime_decorator()
        @rename_function(f.__name__)
        def wrapped(*args, **kwargs):
            logger.slow('Calling %s with arguments %s and %s', f.__name__, args, kwargs)
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            duration = time.perf_counter() - start
            logger.slow('Returning from %s after %s seconds: %s', f.__name__, duration, ret)
            return ret

        return wrapped

    return wrap
