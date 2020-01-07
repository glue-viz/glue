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

    for cid in data.main_components + data.derived_components:

        if components is not None and cid not in components:
            continue

        if data.get_kind(cid) == 'categorical':
            values = data[cid]
            if values.dtype.kind == 'U':
                values = np.char.encode(values, encoding='ascii', errors='replace')
            else:
                values = values.copy()
        else:
            values = data[cid].copy()

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

        f.create_dataset(cid.label, data=values)

    f.close()
