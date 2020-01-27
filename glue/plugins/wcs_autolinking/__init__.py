from distutils.version import LooseVersion

from astropy import __version__


def setup():
    from . import wcs_autolinking  # noqa
