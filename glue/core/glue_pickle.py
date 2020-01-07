from glue.logger import logger

try:
    from dill import dumps, loads  # noqa
except ImportError:
    logger.info("Dill library not installed. Falling back to cPickle")
    from pickle import dumps, loads  # noqa
