from __future__ import absolute_import, division, print_function

from glue.logger import logger

try:
    from dill import dumps, loads
except ImportError:
    logger.info("Dill library not installed. Falling back to cPickle")

from glue.external.six.moves.cPickle import dumps, loads
