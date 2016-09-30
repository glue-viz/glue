from __future__ import absolute_import, division, print_function

import warnings
from os.path import basename
from collections import OrderedDict

from glue.core.coordinates import coordinates_from_header, WCSCoordinates
from glue.core.data import Component, Data
from glue.config import data_factory, qglue_parser


__all__ = ['is_fits', 'fits_reader', 'is_casalike', 'casalike_cube']


def is_fits(filename):
    from astropy.io import fits
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(filename, ignore_missing_end=True):
                return True
    except IOError:
        return False


@data_factory(
    label='FITS file',
    identifier=is_fits,
    priority=100,
)
def fits_reader(source, auto_merge=False, exclude_exts=None, label=None):
    """
    Read in all extensions from a FITS file.

    Parameters
    ----------
    source: str or HDUList
        The pathname to the FITS file.
        If an HDUList is passed in, simply use that.

    auto_merge: bool
        Merge extensions that have the same shape
        and only one has a defined WCS.

    exclude_exts: [hdu, ] or [index, ]
        List of HDU's to exclude from reading.
        This can be a list of HDU's or a list
        of HDU indexes.
    """

    from astropy.io import fits
    from astropy.table import Table

    exclude_exts = exclude_exts or []
    if not isinstance(source, fits.hdu.hdulist.HDUList):
        hdulist = fits.open(source, ignore_missing_end=True)
        hdulist.verify('fix')
    else:
        hdulist = source
    groups = OrderedDict()
    extension_by_shape = OrderedDict()

    if label is not None:

        label_base = label

    else:

        hdulist_name = hdulist.filename()
        if hdulist_name is None:
            hdulist_name = "HDUList"

        label_base = basename(hdulist_name).rpartition('.')[0]

        if not label_base:
            label_base = basename(hdulist_name)

    # Create a new image Data.
    def new_data():
        label = '{0}[{1}]'.format(
            label_base,
            hdu_name
        )
        data = Data(label=label)
        data.coords = coords
        groups[hdu_name] = data
        extension_by_shape[shape] = hdu_name
        return data

    for extnum, hdu in enumerate(hdulist):
        hdu_name = hdu.name if hdu.name else "HDU{0}".format(extnum)
        if (hdu.data is not None and
                hdu.data.size > 0 and
                hdu_name not in exclude_exts and
                extnum not in exclude_exts):
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
                table = Table.read(hdu, format='fits')
                label = '{0}[{1}]'.format(
                    label_base,
                    hdu_name
                )
                data = Data(label=label)
                groups[hdu_name] = data
                for column_name in table.columns:
                    column = table[column_name]
                    if column.ndim != 1:
                        warnings.warn("Dropping column '{0}' since it is not 1-dimensional".format(column_name))
                        continue
                    component = Component.autotyped(column, units=column.unit)
                    data.add_component(component=component,
                                       label=column_name)
    return [groups[idx] for idx in groups]


# Utilities

def is_image_hdu(hdu):
    from astropy.io.fits.hdu import PrimaryHDU, ImageHDU, CompImageHDU
    return isinstance(hdu, (PrimaryHDU, ImageHDU, CompImageHDU))


def is_table_hdu(hdu):
    from astropy.io.fits.hdu import TableHDU, BinTableHDU
    return isinstance(hdu, (TableHDU, BinTableHDU))


def has_wcs(coords):
    return (isinstance(coords, WCSCoordinates) and
               any(axis['coordinate_type'] is not None
               for axis in coords.wcs.get_axis_types()))


def is_casalike(filename, **kwargs):
    """
    Check if a FITS file is a CASA like cube,
    with (P, P, V, Stokes) layout
    """
    from astropy.io import fits

    if not is_fits(filename):
        return False
    with fits.open(filename, ignore_missing_end=True) as hdulist:
        if len(hdulist) != 1:
            return False
        if hdulist[0].header['NAXIS'] != 4:
            return False

        from astropy.wcs import WCS
        w = WCS(hdulist[0].header)

    ax = [a.get('coordinate_type') for a in w.get_axis_types()]
    return ax == ['celestial', 'celestial', 'spectral', 'stokes']


@data_factory(label='CASA PPV Cube', identifier=is_casalike, deprecated=True)
def casalike_cube(filename, **kwargs):
    """
    This provides special support for 4D CASA FITS - like cubes,
    which have 2 spatial axes, a spectral axis, and a stokes axis
    in that order.

    Each stokes cube is split out as a separate component
    """
    from astropy.io import fits

    result = Data()

    if 'ignore_missing_end' not in kwargs:
        kwargs['ignore_missing_end'] = True

    with fits.open(filename, **kwargs) as hdulist:
        array = hdulist[0].data
        header = hdulist[0].header
    result.coords = coordinates_from_header(header)
    for i in range(array.shape[0]):
        result.add_component(array[[i]], label='STOKES %i' % i)
    return result


try:
    from astropy.io.fits import HDUList
except ImportError:
    pass
else:
    # Put HDUList parser before list parser
    @qglue_parser(HDUList, priority=100)
    def _parse_data_hdulist(data, label):
        from glue.core.data_factories.fits import fits_reader
        return fits_reader(data, label=label)
