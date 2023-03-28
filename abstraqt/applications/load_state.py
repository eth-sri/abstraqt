import argparse

from abstraqt.stabilizer.abstract_stabilizer_plus import AbstractStabilizerPlus
from abstraqt.utils import string_helper


string_helper.default_element_wise_str_threshold = 1000_0000


def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('file')
	args = parser.parse_args()

	s = AbstractStabilizerPlus.load_from_file(args.file)

	print(repr(s))


if __name__ == '__main__':
	run()
