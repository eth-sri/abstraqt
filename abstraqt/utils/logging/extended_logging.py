import logging

from abstraqt.utils.function_renamer import rename_function


def add_log_level(number: int, label: str):
    """
    add additional log level
    https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
    https://docs.python.org/3/library/logging.html#levels
    """
    label_lower = label.lower()
    label_upper = label.upper()

    logging.addLevelName(number, label_upper)

    @rename_function(label_lower)
    def log_with_level(self, message, *args, **kws):
        if self.isEnabledFor(number):
            # Yes, logger takes its '*args' as 'args'.
            self._log(number, message, args, **kws)

    setattr(logging.Logger, label_lower, log_with_level)
    setattr(logging, label_upper, number)

    return log_with_level


VERBOSE = 5
verbose = add_log_level(VERBOSE, "VERBOSE")

SLOW = 2
slow = add_log_level(SLOW, "SLOW")
