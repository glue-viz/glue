from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core import Subset
from glue.config import data_exporter


__all__ = []


def make_component_header(component, data_header):
    header = data_header
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

    if isinstance(data, Subset):
        mask = data.to_mask()
        data = data.data
    else:
        mask = None

    data_header = data.coords.header if hasattr(data.coords, "header") else None

    from astropy.io import fits

    hdus = fits.HDUList()

    for cid in data.visible_components:

        if components is not None and cid not in components:
            continue

        comp = data.get_component(cid)
        if comp.categorical:
            # TODO: emit warning
            continue
        else:
            # We need to cast to float otherwise we can't set the masked
            # values to NaN.
            values = comp.data.astype(float, copy=True)

        if mask is not None:
            values[~mask] = np.nan

        # TODO: special behavior for PRIMARY?
        header = make_component_header(comp, data_header) if data_header else None
        hdu = fits.ImageHDU(values, name=cid.label, header=header)
        hdus.append(hdu)

    try:
        hdus.writeto(filename, overwrite=True)
    except TypeError:
        hdus.writeto(filename, clobber=True)
