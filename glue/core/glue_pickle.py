from cPickle import dumps, loads
import logging

try:
    from dill import dumps, loads
except ImportError:
    logging.getLogger(__name__).warn("Dill library not installed. "
                                     "Falling back to cPickle")
