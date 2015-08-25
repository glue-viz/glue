from __future__ import absolute_import, division, print_function

from os.path import basename

from ...config import data_factory
from ..data import Component, Data
from ..coordinates import coordinates_from_header

__all__ = ['fits_container']


@data_factory(
    label='Generic FITS',
    priority=100,
)
def fits_container(source, auto_merge=False, exclude_exts=None, **kwargs):
    """Read in all extensions from a FITS file.

    Parameters
    ----------
    source: str or HDUList
        The pathname to the FITS file.
        If an HDUList is passed in, simply use that.

<<<<<<< HEAD
=======
    auto_merge: bool
        Merge extensions that have the same shape
        and only one has a defined WCS.

>>>>>>> 501e68c... Introduce container readers, starting with FITS
    exclude_exts: [hdu, ] or [index, ]
        List of HDU's to exclude from reading.
        This can be a list of HDU's or a list
        of HDU indexes.
    """
    from ...external.astro import fits
    from astropy.table import Table

    exclude_exts = exclude_exts or []
    if not isinstance(source, fits.hdu.hdulist.HDUList):
        hdulist = fits.open(source)
    else:
        hdulist = source
    groups = dict()
    extension_by_shape = dict()
    label_base = basename(hdulist.filename()).rpartition('.')[0]

    if not label_base:
        label_base = basename(hdulist.filename())

    # Create a new image Data.
    def new_data():
        label = '{}[{}]'.format(
            label_base,
            hdu_name
        )
        data = Data(label=label)
        data.coords = coords
        groups[hdu_name] = data
        extension_by_shape[shape] = hdu_name
        return data

    for extnum, hdu in enumerate(hdulist):
        hdu_name = hdu.name if hdu.name else str(extnum)
        if hdu.data is not None and \
           hdu_name not in exclude_exts and \
           extnum not in exclude_exts:
            if is_image_hdu(hdu):
                shape = hdu.data.shape
                coords = coordinates_from_header(hdu.header)
                if not auto_merge or has_wcs(coords):
                    data = new_data()
                else:
                    try:
                        data = groups[extension_by_shape[shape]]
                    except KeyError:
                        data = new_data()
                data.add_component(component=hdu.data,
                                   label=hdu_name)
            elif is_table_hdu(hdu):
                # Loop through columns and make component list
                table = Table(hdu.data)
                label = '{}[{}]'.format(
                    label_base,
                    hdu_name
                )
                data = Data(label=label)
                groups[hdu_name] = data
                for column_name in table.columns:
                    column = table[column_name]
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


def has_wcs(coords):
    return any(axis['coordinate_type'] is not None
               for axis in coords.wcs.get_axis_types())
