from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from glue.core import Subset
from glue.config import data_exporter


__all__ = []


@data_exporter(label='HDF5', extension=['hdf5'])
def hdf5_writer(filename, data, components=None):
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

    from h5py import File

    f = File(filename, 'w')

    for cid in data.visible_components:

        if components is not None and cid not in components:
            continue

        comp = data.get_component(cid)
        if comp.categorical:
            if comp.labels.dtype.kind == 'U':
                values = np.char.encode(comp.labels, encoding='ascii', errors='replace')
            else:
                values = comp.labels.copy()
        else:
            values = comp.data.copy()

        if mask is not None:
            if values.ndim == 1:
                values = values[mask]
            else:
                if values.dtype.kind == 'f':
                    values[~mask] = np.nan
                elif values.dtype.kind == 'i':
                    values[~mask] = 0
                elif values.dtype.kind == 'S':
                    values[~mask] = ''
                else:
                    warnings.warn("Unknown data type in HDF5 export: {0}".format(values.dtype))
                    continue

        print(values)

        f.create_dataset(cid.label, data=values)

    f.close()
