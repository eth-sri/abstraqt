import os

import pandas as pd

from abstraqt.utils.logging.default_logging import log_directory, now_string
import abstraqt.utils.logging as logging


logger = logging.getLogger(__name__)


def combine_data_logs(parent_directory: str, dtype_hints=None):
	data_logs = {}

	dirs = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
	for d in dirs:
		data_directory = os.path.join(parent_directory, d, 'data')
		if os.path.isdir(data_directory):
			for f in os.listdir(data_directory):
				file_path = os.path.join(data_directory, f)
				if os.path.isfile(file_path) and file_path.endswith('.csv'):
					try:
						df = pd.read_csv(file_path, dtype=dtype_hints)
						if f in data_logs:
							previous = data_logs[f]
							df = pd.concat((previous, df))
						data_logs[f] = df
					except Exception as e:
						logger.warning('Error (see below) when trying to parse %s', file_path)
						logger.exception(e)

	data_summary_dir = os.path.join(parent_directory, 'data_summary_' + now_string)
	os.makedirs(data_summary_dir)
	for f, df in data_logs.items():
		file_path = os.path.join(data_summary_dir, f)
		df.to_csv(file_path, index=False)

	return data_summary_dir


if __name__ == '__main__':
	combine_data_logs(log_directory)
