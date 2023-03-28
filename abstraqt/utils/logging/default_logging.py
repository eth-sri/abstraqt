import logging.handlers as logging_handlers
import os
import shutil
import sys
from datetime import datetime
from logging import getLevelName
from typing import Dict

from appdirs import user_log_dir

from abstraqt.utils.logging.entry_point_helper import log_entry_point
from abstraqt.utils import logging

##############
# FORMATTERS #
##############

full_formatter = logging.Formatter('%(asctime)s [%(levelname)7s, %(name)s]: %(message)s', datefmt="%Y-%m-%d_%H-%M-%S")
standard_formatter = logging.Formatter('[%(levelname)7s, %(name)s] %(message)s')
minimal_formatter = logging.Formatter('%(message)s')


##############################
# CONVENIENCE CONFIGURATIONS #
##############################


class FilterPerName(logging.Filter):
    # https://docs.python.org/3/library/logging.html#filter-objects

    def __init__(self, level_per_name: Dict[str, int]):
        self.level_per_name = level_per_name
        super().__init__()

    def filter(self, log_record: logging.LogRecord):
        for name, level in self.level_per_name.items():
            if log_record.name.startswith(name) and log_record.levelno >= level:
                return True
        return False


def log_to_handler(handler: logging.Handler, level_per_name: Dict[str, int]):
    min_level = min(level_per_name.values())
    handler.setLevel(min_level)

    # filter
    f = FilterPerName(level_per_name)
    handler.addFilter(f)

    logger = logging.getLogger('')
    logger.addHandler(handler)


def prepare_default_logging_stdout():
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(standard_formatter)
    level_per_name = {
        '': logging.WARNING,
        'abstraqt': logging.INFO,
        '__main__': logging.INFO
    }
    log_to_handler(stdout_handler, level_per_name)


bare_log_directory = user_log_dir('abstraqt')

# support sub-directory
log_sub_directory = os.getenv('LOG_SUB_DIRECTORY', 'default')
log_directory = os.path.join(bare_log_directory, log_sub_directory)

now = datetime.now()
now_string = now.strftime("%Y-%m-%d__%H-%M-%S__%f")
process_id = str(os.getpid())
log_directory_prefix = 'log__'
timed_log_directory = os.path.join(log_directory, log_directory_prefix + now_string + '__' + process_id)


def get_fresh_log_file_handler(name, log_level=None):
    os.makedirs(timed_log_directory, exist_ok=True)
    f = os.path.join(timed_log_directory, name)

    file_handler = logging_handlers.RotatingFileHandler(f, maxBytes=10_000_000, backupCount=30, delay=True)
    file_handler.doRollover()
    if log_level:
        file_handler.setLevel(log_level)
    file_handler.setFormatter(full_formatter)

    return file_handler


def configure_log_file_handler(level: int):
    label = getLevelName(level)

    file_handler = get_fresh_log_file_handler(label.lower() + '.log')
    level_per_name = {
        '': logging.INFO,
        'abstraqt': level,
        'tests': level,
        '__main__': level
    }
    log_to_handler(file_handler, level_per_name)

    return file_handler


def prepare_default_logging_files(log_level: str):
    log_level = getLevelName(log_level)
    assert isinstance(log_level, int)

    file_handler = configure_log_file_handler(log_level)

    return file_handler


def enable_default_logging(log_level: str):
    logger = logging.getLogger('')
    logger.setLevel(log_level)

    prepare_default_logging_stdout()
    file_handler = prepare_default_logging_files(log_level)

    logger = logging.getLogger(__name__)

    logger.info('Enabled default logging to %s at %s', file_handler.baseFilename, now_string)

    log_entry_point(logger)


default_logging = os.environ.get('DEFAULT_LOGGING')
is_logging_enabled = default_logging is not None


def _enable_default_logging_on_flag():
    if is_logging_enabled:
        log_level = str(default_logging).upper()
        enable_default_logging(log_level)
