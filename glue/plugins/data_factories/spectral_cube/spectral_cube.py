from __future__ import absolute_import, division, print_function

from spectral_cube import StokesSpectralCube

from glue.core import Data
from glue.config import data_factory
from glue.core.data_factories.fits import is_fits
from glue.core.coordinates import coordinates_from_wcs


def is_spectral_cube(filename, **kwargs):
    """
    Check that the file is a 3D or 4D FITS spectral cube
    """

    if not is_fits(filename):
        return False

    try:
        StokesSpectralCube.read(filename)
    except Exception:
        return False
    else:
        return True


@data_factory(label='FITS Spectral Cube', identifier=is_spectral_cube)
def spectral_cube_reader(filename, **kwargs):
    """
    Read in a FITS spectral cube. If multiple Stokes components are present,
    these are split into separate glue components.
    """

    cube = StokesSpectralCube.read(filename)

    result = Data()
    result.coords = coordinates_from_wcs(cube.wcs)

    for component in cube.components:
        data = getattr(cube, component).unmasked_data[...]
        result.add_component(data, label='STOKES {0}'.format(component))

    return result
