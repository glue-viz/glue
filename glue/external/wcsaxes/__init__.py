# Licensed under a 3-clause BSD style license - see LICENSE.rst

from distutils.version import LooseVersion
from astropy import __version__
if LooseVersion(__version__) < LooseVersion('0.3'):
    raise ImportError("wcsaxes requires astropy >= 0.3")

from .core import *
from .coordinate_helpers import CoordinateHelper
from .coordinates_map import CoordinatesMap
from .wcs_wrapper import WCS
