from __future__ import absolute_import, division, print_function

import logging

try:
    from dill import dumps, loads
except ImportError:
    logging.getLogger(__name__).warn("Dill library not installed. "
                                     "Falling back to cPickle")

from glue.external.six.moves.cPickle import dumps, loads
