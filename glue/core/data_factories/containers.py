from __future__ import absolute_import, division, print_function

from os.path import basename

from astropy.table import Table

from ...external.astro import fits
from ...config import data_factory
from ..data import Component, Data
from ..coordinates import coordinates_from_header

__all__ = ['fits_container']


@data_factory('Generic FITS', )
def fits_container(source, exclude_exts=None, **kwargs):
    """Read in all extensions from a FITS file.

    Parameters
    ----------
    source: str or HDUList
        The pathname to the FITS file.
        If and HDUList is passed in, simply use that.

    exclude_exts: [hdu, ] or [index, ]
        List of HDU's to exclude from reading.
        This can be a list of HDU's or a list
        of HDU indexes.
    """
    exclude_exts = exclude_exts or []
    if not isinstance(source, fits.hdu.hdulist.HDUList):
        hdulist = fits.open(source)
    else:
        hdulist = source
    groups = dict()
    label_base = basename(hdulist.filename()).rpartition('.')[0]

    if not label_base:
        label_base = basename(hdulist.filename())

    for extnum, hdu in enumerate(hdulist):
        hdu_name = hdu.name if hdu.name else str(extnum)
        if hdu.data is not None and \
           hdu_name not in exclude_exts and \
           extnum not in exclude_exts:
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
                    component = Component(column, units=column.unit)
                    data.add_component(component=component,
                                       label=column_name)
    return groups.values()


# Utilities
def is_image_hdu(hdu):
    from astropy.io.fits.hdu import PrimaryHDU, ImageHDU
    return isinstance(hdu, (PrimaryHDU, ImageHDU))


def is_table_hdu(hdu):
    from astropy.io.fits.hdu import TableHDU, BinTableHDU
    return isinstance(hdu, (TableHDU, BinTableHDU))
