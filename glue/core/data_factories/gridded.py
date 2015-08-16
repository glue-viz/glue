from __future__ import absolute_import, division, print_function

from ..data import Component, Data
from .io import extract_data_fits, extract_data_hdf5
from ...utils import file_format
from ..coordinates import coordinates_from_header

from .helpers import set_default_factory, __factories__

__all__ = ['is_casalike', 'gridded_data', 'casalike_cube']


def is_hdf5(filename):
    # All hdf5 files begin with the same sequence
    with open(filename, 'rb') as infile:
        return infile.read(8) == b'\x89HDF\r\n\x1a\n'


def is_fits(filename):
    from ...external.astro import fits
    try:
        with fits.open(filename):
            return True
    except IOError:
        return False


def gridded_data(filename, format='auto', wcs_ext=0, **kwargs):
    """
    Construct an n - dimensional data object from ``filename``. If the
    format cannot be determined from the extension, it can be
    specified using the ``format`` option. Valid formats are 'fits' and
    'hdf5'.

    :param wcs_ext:
        Specify the number of the extension used to read the wcs info.
    :param kwargs:
        Extra arguments passed to the function used to extract data
        (``extract_data_fits`` or ``extract_data_hdf5``).

    """
    result = Data()

    # Try and automatically find the format if not specified
    if format == 'auto':
        format = file_format(filename)

    # Read in the data
    if is_fits(filename):
        from ...external.astro import fits
        arrays = extract_data_fits(filename, **kwargs)
        header = fits.getheader(filename, ext=wcs_ext)
        result.coords = coordinates_from_header(header)
    elif is_hdf5(filename):
        arrays = extract_data_hdf5(filename, **kwargs)
    else:
        raise Exception("Unkonwn format: %s" % format)

    for component_name in arrays:
        comp = Component.autotyped(arrays[component_name])
        result.add_component(comp, component_name)
    return result


def is_gridded_data(filename, **kwargs):
    if is_hdf5(filename):
        return True

    from ...external.astro import fits
    if is_fits(filename):
        with fits.open(filename) as hdulist:
            for hdu in hdulist:
                if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                    return False
            return True
    return False


gridded_data.label = "FITS/HDF5 Image"
gridded_data.identifier = is_gridded_data
__factories__.append(gridded_data)
set_default_factory('fits', gridded_data)
set_default_factory('hd5', gridded_data)
set_default_factory('hdf5', gridded_data)


def casalike_cube(filename, **kwargs):
    """
    This provides special support for 4D CASA - like cubes,
    which have 2 spatial axes, a spectral axis, and a stokes axis
    in that order.

    Each stokes cube is split out as a separate component
    """
    from ...external.astro import fits

    result = Data()
    with fits.open(filename, **kwargs) as hdulist:
        array = hdulist[0].data
        header = hdulist[0].header
    result.coords = coordinates_from_header(header)
    for i in range(array.shape[0]):
        result.add_component(array[[i]], label='STOKES %i' % i)
    return result


def is_casalike(filename, **kwargs):
    """
    Check if a file is a CASA like cube,
    with (P, P, V, Stokes) layout
    """
    from ...external.astro import fits

    if not is_fits(filename):
        return False
    with fits.open(filename) as hdulist:
        if len(hdulist) != 1:
            return False
        if hdulist[0].header['NAXIS'] != 4:
            return False

        from astropy.wcs import WCS
        w = WCS(hdulist[0].header)

    ax = [a.get('coordinate_type') for a in w.get_axis_types()]
    return ax == ['celestial', 'celestial', 'spectral', 'stokes']


casalike_cube.label = 'CASA PPV Cube'
casalike_cube.identifier = is_casalike
__factories__.append(casalike_cube)
