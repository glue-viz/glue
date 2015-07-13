from __future__ import absolute_import, division, print_function

from os.path import basename

from astropy.table import Table

from ..data import Component, Data
from ..coordinates import coordinates_from_header
from ...external.astro import fits

from .helpers import set_default_factory, __factories__
from .gridded import is_fits

__all__ = ['fits_container']


def fits_container(filename, **kwargs):
    """
    Read in all possible extensions from a FITS file.
    Extensions are group by dimensionality of data,
    including table extensions.
    """
    hdulist = fits.open(filename)
    groups = dict()
    label_base = basename(filename).rpartition('.')[0]
    if not label_base:
        label_base = basename(filename)
    for extnum, hdu in enumerate(hdulist):
        if hdu.data is not None:
            hdu_name = hdu.name if hdu.name else str(extnum)
            if is_image_hdu(hdu):
                shape = hdu.data.shape
                try:
                    data = groups[shape]
                except KeyError:
                    label = '{}[{}]'.format(
                        label_base,
                        'x'.join(str(x) for x in shape)
                    )
                    data = Data(label=label)
                    data.coords = coordinates_from_header(hdu.header)
                    groups[shape] = data
                data.add_component(component=hdu.data,
                                   label=hdu_name)
            elif is_table_hdu(hdu):
                # Loop through columns and make component list
                table = Table(hdu.data)
                table_name = '{}[{}]'.format(
                    label_base,
                    hdu_name
                )
                for column_name in table.columns:
                    column = table[column_name]
                    shape = column.shape
                    data_label = '{}[{}]'.format(
                        table_name,
                        'x'.join(str(x) for x in shape)
                    )
                    try:
                        data = groups[data_label]
                    except KeyError:
                        data = Data(label=data_label)
                        groups[data_label] = data
                    component = Component.autotyped(column, units=column.unit)
                    data.add_component(component=component,
                                       label=column_name)
    return [data for data in groups.itervalues()]


fits_container.label = "FITS full file"
fits_container.identifier = is_fits
__factories__.append(fits_container)
set_default_factory('fits', fits_container)


# Utilities
def is_image_hdu(hdu):
    from astropy.io.fits.hdu import PrimaryHDU, ImageHDU
    return reduce(lambda x, y: x | isinstance(hdu, y), (PrimaryHDU, ImageHDU), False)


def is_table_hdu(hdu):
    from astropy.io.fits.hdu import TableHDU, BinTableHDU
    return reduce(lambda x, y: x | isinstance(hdu, y), (TableHDU, BinTableHDU), False)
