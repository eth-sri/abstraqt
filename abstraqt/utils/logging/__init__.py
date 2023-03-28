from logging import *

from .default_logging import _enable_default_logging_on_flag
from .extended_logging import SLOW, slow
from .extended_logging import VERBOSE, verbose

# See also https://docs.python.org/3/howto/logging-cookbook.html for proper logging


_enable_default_logging_on_flag()
