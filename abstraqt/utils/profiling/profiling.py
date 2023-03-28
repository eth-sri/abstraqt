import os
import io

import cProfile
import pstats

from abstraqt.utils import logging
from abstraqt.utils.logging.default_logging import is_logging_enabled
from abstraqt.utils.logging.log_data import data_dir
from abstraqt.utils.string_helper import get_environment_variable_bool


logger = logging.getLogger(__name__)
should_profile = get_environment_variable_bool('ABSTRAQT_PROFILE', default=False) and is_logging_enabled
if should_profile:
	logger.info('Storing profiling information to %s', data_dir)


profiler_running = False


class profile:

	def __init__(self, label: str):
		self.label = label
		self.profiler = None

	def __enter__(self):
		if should_profile:
			self.profiler = cProfile.Profile()
			self.profiler.enable()

			global profiler_running
			if profiler_running:
				raise ValueError('Profiler was aady started')
			profiler_running = True

	def __exit__(self, type, value, traceback):
		if should_profile:
			global profiler_running
			assert profiler_running
			profiler_running = False

			self.profiler.disable()

			profile_file = os.path.join(data_dir, self.label)
			logger.debug('Storing profile information to %s', profile_file)

			# write parse-able text file
			s = io.StringIO()
			ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumtime')
			ps.print_stats()

			# dump to file
			stats = pstats.Stats(self.profiler).sort_stats('cumtime')
			stats.dump_stats(profile_file + '.pstats')

			with open(profile_file + '.profile', 'w+') as f:
				f.write(s.getvalue())

			# # write to console
			# stats = pstats.Stats(self.profiler).sort_stats('cumtime')
			# stats.print_stats()
