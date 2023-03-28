import os
from typing import Callable, Dict, IO, Union
import atexit
import time

from abstraqt.utils import logging
from abstraqt.utils.function_renamer import rename_function
from abstraqt.utils.logging.default_logging import timed_log_directory, is_logging_enabled
from abstraqt.utils.string_helper import get_environment_variable_bool


logger = logging.getLogger(__name__)
should_log_data = get_environment_variable_bool('ABSTRAQT_LOG_DATA', default=True) and is_logging_enabled

data_dir = os.path.join(timed_log_directory, 'data')
os.makedirs(data_dir, exist_ok=True)


data_context_dir = {}


class set_data_context:

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def __enter__(self):
		for key in self.args:
			self._check_key(key)
			data_context_dir[key] = None

		for key, value in self.kwargs.items():
			self._check_key(key)
			data_context_dir[key] = value

	@staticmethod
	def _check_key(key):
		if key in data_context_dir:
			raise ValueError(f'Key {key} is already in the data context')

	def __exit__(self, type, value, traceback):
		for key in self.args:
			del data_context_dir[key]
		for key in self.kwargs.keys():
			del data_context_dir[key]


def set_data_context_decorator(label: str = None):
	def wrap(f: Callable):

		if label is None:
			label_ = f.__name__
		else:
			label_ = label

		@rename_function(f.__name__)
		def wrapped(*args, **kwargs):
			with set_data_context(label_):
				return f(*args, **kwargs)

		return wrapped

	return wrap


data_writers: Dict[str, IO] = {}


def close_all_data_writers():
	for k, v in data_writers.items():
		logger.debug('Closing data logger %s', k)
		v.close()


atexit.register(close_all_data_writers)


def get_data_writer(label: str, first_line: Union[str, Callable[[], str]] = None):
	data_writer = data_writers.get(label)
	if data_writer is None:
		data_directory = os.path.join(timed_log_directory, 'data')
		os.makedirs(data_directory, exist_ok=True)
		f = os.path.join(data_directory, label + '.csv')

		data_writer = open(f, 'w')
		data_writers[label] = data_writer

		if first_line is not None:
			if not isinstance(first_line, str):
				first_line = first_line()
			data_writer.write(first_line + '\n')

	return data_writer


def log_data(**data):
	if should_log_data:
		all_keys = '__'.join(data_context_dir.keys()) + '__' + '__'.join(data.keys())

		def columns():
			return ','.join(data_context_dir.keys()) + ',' + ','.join(data.keys())

		data_writer = get_data_writer(all_keys, columns)
		msg = ','.join(str(x) for x in data_context_dir.values()) + ',' + ','.join(str(x) for x in data.values())
		data_writer.write(msg + '\n')


def log_data_without_context(label: str, value):
	if should_log_data:
		data_writer = get_data_writer(label, label)
		data_writer.write(str(value) + '\n')


###########
# RUNTIME #
###########


max_overhead_dict: Dict[str, float] = {}
total_measured_time_dict: Dict[str, float] = {}
total_log_time_dict: Dict[str, float] = {}


def _log_runtime(label: str, start: float, report_overhead=True):
	if should_log_data:
		# measure time
		t = time.perf_counter() - start
		total_measured_time = total_measured_time_dict.get(label, 0)
		total_measured_time += t
		total_measured_time_dict[label] = total_measured_time

		# also measure time for logging (can be slow in some cases)
		log_start = time.perf_counter()

		# log
		log_data(**{label + '_runtime': t})
		# log_data(**{self.label + '_runtime': t})

		if report_overhead:
			max_overhead = max_overhead_dict.get(label, 0.1)

			# warn if the logging overhead is too high
			total_log_time = total_log_time_dict.get(label, 0)
			log_time = time.perf_counter() - log_start
			total_log_time += log_time
			total_log_time_dict[label] = total_log_time

			log_percentage = total_log_time / total_measured_time
			if log_percentage > max_overhead:
				max_overhead_dict[label] = log_percentage
				logger.warning('Logging took %.1f%% of the runtime for %s', log_percentage * 100, label)


class report_runtime:

	def __init__(self, label: str):
		self.label = label
		self.start = 0

	def __enter__(self):
		self.start = time.perf_counter()

	def __exit__(self, exc_type, exc_val, exc_tb):
		_log_runtime(self.label, self.start)


def report_runtime_decorator(label: str = None, report_overhead=True):
	def wrap(f: Callable):

		if label is None:
			label_ = f.__name__
		else:
			label_ = label

		@rename_function(f.__name__)
		def wrapped(*args, **kwargs):
			start = time.perf_counter()
			ret = f(*args, **kwargs)
			_log_runtime(label_, start, report_overhead=report_overhead)
			return ret
		return wrapped
	return wrap
