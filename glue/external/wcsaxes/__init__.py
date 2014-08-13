from distutils.version import LooseVersion
from astropy import __version__
if LooseVersion(__version__) < LooseVersion('0.3'):
    raise ImportError("wcsaxes requires astropy >= 0.3")

from .wcsaxes import *
from .coordinate_helpers import CoordinateHelper
