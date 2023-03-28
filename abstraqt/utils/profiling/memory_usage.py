import resource
from sys import platform


def get_ru_maxrss():
    """
    maximum kilobytes used by this process, in its lifetime, in KiB (kibibytes)
    """
    # https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    ret = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if platform == "darwin":
        # Mac reports in bytes
        ret /= 1024

    return ret


if __name__ == '__main__':
    import numpy as np
    n = 1000 * 1000 * 1000
    x = np.full(n, 20, dtype=np.int32)

    # allocates n * 32 bits, which is ~4_000_000 kB
    print('Expected: 3_906_000')

    r = get_ru_maxrss()
    print('Actual:', r)
