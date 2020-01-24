import numpy as np

from astropy.wcs import WCS

from glue.core import Subset, Data
from glue.config import data_exporter

__all__ = []


def make_component_header(component, header):
    """
    Function that extracts information from components
    and adds it to the data header. The input header is
    expected to come from Data.coords.header by default.
    Parameters
    ----------
    component: glue Component
        Glue component to extract info from
    header: astropy.io.fits.header.Header
        Input header to be modified according to
        the input component
    """

    # Add units information
    header["BUNIT"] = component.units

    return header


@data_exporter(label='FITS (1 component/HDU)', extension=['fits', 'fit'])
def fits_writer(filename, data, components=None):
    """
    Write a dataset or a subset to a FITS file.

    Parameters
    ----------
    data : `~glue.core.data.Data` or `~glue.core.subset.Subset`
        The data or subset to export
    components : `list` or `None`
        The components to export. Set this to `None` to export all components.
    """

    from astropy.io import fits

    if isinstance(data, Subset):
        mask = data.to_mask()
        data = data.data
    else:
        mask = None

    data_header = data.coords.to_header() if isinstance(data.coords, WCS) else fits.Header()

    hdus = fits.HDUList()

    for cid in data.main_components + data.derived_components:

        if components is not None and cid not in components:
            continue

        if data.get_kind(cid) != 'numerical':
            # TODO: emit warning
            continue

        values = data[cid]

        if mask is not None:
            # We need to copy the values so that we can mask them
            values = values.copy()
            if values.dtype.kind == 'f':
                blank = None
                values[~mask] = np.nan
            elif values.dtype.kind == 'i':
                blank = np.iinfo(values.dtype).min
                values[~mask] = blank

        # TODO: special behavior for PRIMARY?
        if isinstance(data, Data):
            comp = data.get_component(cid)
            header = make_component_header(comp, data_header)
        else:
            header = fits.Header()

        if mask is not None and blank is not None:
            header['BLANK'] = blank

        hdu = fits.ImageHDU(values, name=cid.label, header=header)
        hdus.append(hdu)

    try:
        hdus.writeto(filename, overwrite=True)
    except TypeError:
        hdus.writeto(filename, clobber=True)
