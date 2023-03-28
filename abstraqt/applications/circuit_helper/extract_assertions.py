import re
from typing import Iterable


bases_assertion_marker = '// assert bases '


def extract_assertions_from_file(circuit_file: str):
    with open(circuit_file, "r") as f:
        return extract_assertions_from_lines(f)


def extract_assertions_from_lines(lines: Iterable[str]):
    ret = None
    for line in lines:
        if line.startswith(bases_assertion_marker):
            if ret is None:
                ret = re.findall(r'\|([^>]*)>', line)
            else:
                raise ValueError('Found multiple assertions.')
    return ret


def extract_assertions_from_line(line: str):
    ret = re.findall(r'\|([^>]*)>', line)
    return ret
