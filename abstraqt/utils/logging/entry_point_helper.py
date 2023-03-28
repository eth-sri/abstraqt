import os
import subprocess
import sys


def log_entry_point(logger):
    h = get_git_revision_short_hash()
    if h is not None:
        logger.debug('Running git commit hash %s', h)
    logger.debug('Command line arguments passed to Python: %s', sys.argv)
    logger.debug('Environment variables: %s', os.environ)


def get_git_revision_short_hash():
    script_directory = os.path.dirname(os.path.realpath(__file__))
    try:
        ret = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=script_directory)
        ret = ret.decode('ascii').strip()
        return ret
    except subprocess.CalledProcessError:
        return None
