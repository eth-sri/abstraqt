import sys
import traceback


def get_exception_information():
    _, _, tb = sys.exc_info()
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    return filename, line, func, text
