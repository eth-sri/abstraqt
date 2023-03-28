import traceback


def get_stack_trace():
    lines = [line.strip() for line in traceback.format_stack()]
    return lines
