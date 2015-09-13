# NOTE: This file exists only for the purposes of backward-compatibility

from __future__ import absolute_import, division, print_function

from ..data import Component, Data
from ...utils import file_format
from ..coordinates import coordinates_from_header
from ...config import data_factory

from .fits import is_fits, extract_data_fits, is_casalike, casalike_cube
from .hdf5 import is_hdf5, extract_data_hdf5

__all__ = ['is_gridded_data', 'gridded_data', 'is_casalike', 'casalike_cube']


def is_gridded_data(filename, **kwargs):

    if is_hdf5(filename):
        return True

    if is_fits(filename):
        from ...external.astro import fits
        with fits.open(filename) as hdulist:
            return is_image_hdu(hdulist[0])

    return False


@data_factory(label="FITS/HDF5 Image", identifier=is_gridded_data, deprecated=True)
def gridded_data(filename, format='auto', **kwargs):

    result = Data()

    # Try and automatically find the format if not specified
    if format == 'auto':
        format = file_format(filename)

    # Read in the data
    if is_fits(filename):
        from ...external.astro import fits
        arrays = extract_data_fits(filename, **kwargs)
        header = fits.getheader(filename)
        result.coords = coordinates_from_header(header)
    elif is_hdf5(filename):
        arrays = extract_data_hdf5(filename, **kwargs)
    else:
        raise Exception("Unkonwn format: %s" % format)

    for component_name in arrays:
        comp = Component.autotyped(arrays[component_name])
        result.add_component(comp, component_name)

    return result
