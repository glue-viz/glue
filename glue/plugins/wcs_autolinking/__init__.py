from distutils.version import LooseVersion

from astropy import __version__

# The autolinking functionality relies on the new WCS API present in
# Astropy 3.1 and later.
ASTROPY_GE_31 = LooseVersion(__version__) >= '3.1'


def setup():

    if not ASTROPY_GE_31:
        return

    from . import wcs_autolinking  # noqa
