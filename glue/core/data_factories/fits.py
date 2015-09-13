from __future__ import absolute_import, division, print_function

from os.path import basename

from ...compat.collections import OrderedDict
from ...config import data_factory
from ..data import Component, Data
from ..coordinates import coordinates_from_header

__all__ = ['fits_reader']


def is_fits(filename):
    from ...external.astro import fits
    try:
        with fits.open(filename):
            return True
    except IOError:
        return False


def filter_hdulist_by_shape(hdulist, use_hdu='all'):
    """
    Remove empty HDUs, and ensure that all HDUs can be
    packed into a single Data object (ie have the same shape)

    Parameters
    ----------
    use_hdu : 'all' or list of integers (optional)
        Which HDUs to use

    Returns
    -------
    a new HDUList
    """
    from ...external.astro import fits

    # If only a subset are requested, extract those
    if use_hdu != 'all':
        hdulist = [hdulist[hdu] for hdu in use_hdu]

    # Now only keep HDUs that are not tables or empty.
    valid_hdus = []
    for hdu in hdulist:
        if (isinstance(hdu, fits.PrimaryHDU) or
                isinstance(hdu, fits.ImageHDU)) and \
                hdu.data is not None:
            valid_hdus.append(hdu)

    # Check that dimensions of all HDU are the same
    # Allow for HDU's that have no data.
    reference_shape = valid_hdus[0].data.shape
    for hdu in valid_hdus:
        if hdu.data.shape != reference_shape:
            raise Exception("HDUs are not all the same dimensions")

    return valid_hdus


def extract_data_fits(filename, use_hdu='all'):
    '''
    Extract non-tabular HDUs from a FITS file. If `use_hdu` is 'all', then
    all non-tabular HDUs are extracted, otherwise only the ones specified
    by `use_hdu` are extracted (`use_hdu` should then contain a list of
    integers). If the requested HDUs do not have the same dimensions, an
    Exception is raised.
    '''
    from ...external.astro import fits

    # Read in all HDUs
    hdulist = fits.open(filename, ignore_blank=True)
    hdulist = filter_hdulist_by_shape(hdulist)

    # Extract data
    arrays = {}
    for hdu in hdulist:
        arrays[hdu.name] = hdu.data

    return arrays


@data_factory(
    label='FITS file',
    identifier=is_fits,
    priority=100,
)
def fits_reader(source, auto_merge=False, exclude_exts=None):
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

    from ...external.astro import fits
    from astropy.table import Table

    exclude_exts = exclude_exts or []
    if not isinstance(source, fits.hdu.hdulist.HDUList):
        hdulist = fits.open(source)
    else:
        hdulist = source
    groups = OrderedDict()
    extension_by_shape = OrderedDict()

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
        hdu_name = hdu.name if hdu.name else str(extnum)
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
                table = Table(hdu.data)
                label = '{0}[{1}]'.format(
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
    return [groups[idx] for idx in groups]


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


@data_factory(label='CASA PPV Cube', identifier=is_casalike)
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
