import os

from abstraqt.utils import logging
from abstraqt.utils.cache_dir import cache_dir
from abstraqt.utils.hash_helper import my_hash

cachier_cache_directory = os.path.join(cache_dir, 'cachier')
logger = logging.getLogger(__name__)


def hash_params(*args, **kwargs):
    h = my_hash((args, kwargs))
    logger.slow('Hashed %s %s to %s', args, kwargs, h)
    return h


default_cachier_arguments = {
    'pickle_reload': False,
    'cache_dir': cachier_cache_directory,
    'hash_params': hash_params
}
