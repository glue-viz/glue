import logging
from glue.external.six.moves.cPickle import dumps, loads

try:
    from dill import dumps, loads
except ImportError:
    logging.getLogger(__name__).warn("Dill library not installed. "
                                     "Falling back to cPickle")
