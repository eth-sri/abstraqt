from typing import Sequence

import numpy as np


def is_equal(x, y):
    if hasattr(x, 'representation'):
        x = getattr(x, 'representation')
    if hasattr(y, 'representation'):
        y = getattr(y, 'representation')

    same = np.isclose(x, y)

    if isinstance(same, np.ndarray):
        same = np.all(same)

    return same


def filter_equal(xs: Sequence):
    ret = []
    for x in xs:
        already = False
        for y in ret:
            if is_equal(x, y):
                already = True

        if already:
            ret.append(x)

    return ret
