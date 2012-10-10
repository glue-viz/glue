# Set up configuration variables
try:
    from sip import setapi
except ImportError:
    pass
else:
    setapi('QString', 2)
    setapi('QVariant', 2)

import logging
logging.basicConfig(level=logging.WARNING)

from .config import load_configuration
env = load_configuration()

from .version import __version__
