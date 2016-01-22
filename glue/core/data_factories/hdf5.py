from __future__ import absolute_import, division, print_function

import os
import warnings
from collections import OrderedDict

from glue.core.data import Component, Data
from glue.config import data_factory


__all__ = ['is_hdf5', 'hdf5_reader']


def extract_hdf5_datasets(handle):
    '''
    Recursive function that returns a dictionary with all the datasets
    found in an HDF5 file or group. `handle` should be an instance of
    h5py.highlevel.File or h5py.highlevel.Group.
    '''

    import h5py

    datasets = {}
    for group in handle:
        if isinstance(handle[group], h5py.highlevel.Group):
            sub_datasets = extract_hdf5_datasets(handle[group])
            for key in sub_datasets:
                datasets[key] = sub_datasets[key]
        elif isinstance(handle[group], h5py.highlevel.Dataset):
            if handle[group].dtype.kind in ('f', 'i', 'V'):
                datasets[handle[group].name] = handle[group]
    return datasets


def is_hdf5(filename):
    # All hdf5 files begin with the same sequence
    with open(filename, 'rb') as infile:
        return infile.read(8) == b'\x89HDF\r\n\x1a\n'


@data_factory(label="HDF5 file", identifier=is_hdf5, priority=100)
def hdf5_reader(filename, format='auto', auto_merge=False, **kwargs):
    """
    Read in all datasets from an HDF5 file

    Parameters
    ----------
    source: str or HDUList
        The pathname to the FITS file.
        If an HDUList is passed in, simply use that.
    """

    import h5py
    from astropy.table import Table

    # Open file
    file_handle = h5py.File(filename, 'r')

    # Define function to read

    # Read in all datasets
    datasets = extract_hdf5_datasets(file_handle)

    label_base = os.path.basename(filename).rpartition('.')[0]

    if not label_base:
        label_base = os.path.basename(filename)

    data_by_shape = {}

    groups = OrderedDict()

    for key in datasets:
        label = '{0}[{1}]'.format(
            label_base,
            key
        )
        if datasets[key].dtype.kind in ('f', 'i'):
            if auto_merge and datasets[key].value.shape in data_by_shape:
                data = data_by_shape[datasets[key].value.shape]
            else:
                data = Data(label=label)
                data_by_shape[datasets[key].value.shape] = data
                groups[label] = data
            data.add_component(component=datasets[key].value, label=key)
        else:
            table = Table.read(datasets[key], format='hdf5')
            data = Data(label=label)
            groups[label] = data
            for column_name in table.columns:
                column = table[column_name]
                if column.ndim == 1:
                    component = Component(column, units=column.unit)
                    data.add_component(component=component,
                                       label=column_name)
                else:
                    warnings.warn("HDF5: Ignoring vector column {0}".format(column_name))

    # Close HDF5 file
    file_handle.close()

    return [groups[idx] for idx in groups]
