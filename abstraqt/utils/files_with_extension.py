import os

from tqdm import tqdm

from abstraqt.utils.string_helper import shorten_string, replace_from_right


def get_files_with_extension(directory: str, extension: str):
	assert os.path.isdir(directory)

	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith(extension):
				full_file = os.path.join(root, file)
				yield full_file


def get_files_with_extension_progress_bar(directory: str, extension: str):
	ret = list(get_files_with_extension(directory, extension))
	t = tqdm(ret)
	for f in t:
		basename = os.path.basename(f)
		basename = replace_from_right(basename, extension, '')
		description = shorten_string(basename, 15)
		t.set_description(description)
		yield f
