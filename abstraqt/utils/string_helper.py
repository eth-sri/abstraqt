import os
from typing import Dict

import numpy as np

# Warning: Changing this value may significantly impact the performance of __str__ and therefore of logging
default_element_wise_str_threshold = 50


def dict_to_string(d: Dict):
    with np.printoptions(threshold=default_element_wise_str_threshold):
        entries = [str(k) + ':\n' + str(v) for k, v in d.items()]
        entries = '\n\n'.join(entries)
        return entries


def replace_from_right(s: str, old, new, n_replacements=1):
    # inspired by https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
    splits = s.rsplit(old, n_replacements)
    return new.join(splits)


def shorten_string(s: str, target_length: int, ellipsis='...'):
    length = len(s)
    if length <= target_length:
        return s.ljust(target_length)
    else:
        n_ellipsis = len(ellipsis)
        assert target_length >= n_ellipsis
        start = (target_length - n_ellipsis) // 2
        left = s[:start]
        end = target_length - n_ellipsis - start
        if end > 0:
            right = s[-end:]
        else:
            right = ''
        return left + ellipsis + right


def time_to_human_readable(seconds: float):
    if seconds < 60:
        return number_to_human_readable(seconds) + 's'
    minutes = seconds / 60
    if minutes < 60:
        return number_to_human_readable(minutes) + 'm'
    hours = minutes / 60
    if hours < 60:
        return number_to_human_readable(hours) + 'h'
    days = hours / 24
    if days < 7:
        return number_to_human_readable(days) + 'd'
    weeks = days / 7
    return number_to_human_readable(weeks)


def number_to_human_readable(x: float):
    if x < 9:
        return '%.1f' % x
    else:
        return '%.0f' % x


def string_to_bool(s: str, default: bool):
    if s in ['true', '1', 't', 'y', 'yes']:
        return True
    elif s in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        return default


def get_environment_variable_bool(key: str, default: bool):
    ret = os.environ.get(key)
    ret = string_to_bool(ret, default=default)
    return ret
