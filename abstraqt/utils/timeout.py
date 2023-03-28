import signal
import traceback


class TimedOutException(Exception):
    pass


def handler(signum, frame):
    assert signum == signal.SIGALRM
    stack = traceback.format_stack(frame)
    deepest = stack[-1]
    raise TimedOutException('Timed out in: ' + deepest + '\nwith stack\n\n' + ''.join(stack))


signal.signal(signal.SIGALRM, handler)


class timeout_after:

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        seconds_before_next_alarm = signal.alarm(self.seconds)
        if seconds_before_next_alarm != 0:
            raise ValueError('Trying to set multiple alarms; cannot nest "abort_after"')

    def __exit__(self, exc_type, exc_val, exc_tb):
        # reset existing signals
        signal.alarm(0)
