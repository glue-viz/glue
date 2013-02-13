# Set up configuration variables

try:
    from sip import setapi
except ImportError:
    pass
else:
    setapi('QString', 2)
    setapi('QVariant', 2)

import logging
try:
    from logging import NullHandler
except ImportError:  # python 2.6 workaround
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger('glue').addHandler(NullHandler())

from .config import load_configuration
env = load_configuration()

from .version import __version__
