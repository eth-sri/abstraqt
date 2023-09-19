import contextlib
import sys
import io


@contextlib.contextmanager
def redirect_stderr_to_string():
    original_stderr = sys.stderr
    stderr_buffer = io.StringIO()
    sys.stderr = stderr_buffer
    try:
        yield
    finally:
        sys.stderr = original_stderr
