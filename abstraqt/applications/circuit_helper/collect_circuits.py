import os

from tqdm import tqdm

from abstraqt.applications.circuit_helper.qiskit_helper import load_from_qasm_file
from abstraqt.utils import logging
from abstraqt.utils.files_with_extension import get_files_with_extension
from abstraqt.utils.string_helper import replace_from_right, shorten_string

logger = logging.getLogger(__name__)


def _collect_circuit_files(directory, extension=None):
    logger.debug('Collecting circuit files in %s', directory)

    if extension is None:
        extension = '.qasm'

    return get_files_with_extension(directory, extension)


def collect_circuit_files(directory, extension=None, max_circuits=None):
    files = list(_collect_circuit_files(directory, extension=extension))
    # make paths relative to current working directory
    files = [os.path.relpath(f, os.getcwd()) for f in files]
    # sort files by size, to start with simpler examples
    files = sorted(files, key=lambda f: os.stat(f).st_size)
    if max_circuits is not None:
        files = files[:max_circuits]
    return files


def short_circuit_description(circuit_file, shorten=True):
    description = os.path.basename(circuit_file)
    description = replace_from_right(description, '.qasm', '')
    if shorten:
        description = shorten_string(description, 15)
    return description


def collect_circuits(directory, extension=None, max_circuits=None):
    t = tqdm(collect_circuit_files(directory, extension=extension, max_circuits=max_circuits))
    for circuit_file in t:
        description = short_circuit_description(circuit_file)
        t.set_description(description)
        circuit = load_from_qasm_file(circuit_file, relative_to_directory=directory)
        yield circuit_file, circuit


def collect_circuits_helper(circuit_file=None, directory=None, from_directory=False):
    if from_directory:
        assert directory is not None
        for circuit_file, circuit in collect_circuits(directory):
            yield circuit_file, circuit
    else:
        assert circuit_file is not None

        circuit = load_from_qasm_file(circuit_file)
        yield circuit_file, circuit
