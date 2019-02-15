from distutils.version import LooseVersion

from astropy import __version__


def setup():

    # The autolinking functionality relies on the new WCS API present in
    # Astropy 3.1 and later.
    if LooseVersion(__version__) < '3.1':
        return

    from . import wcs_autolinking  # noqa
