from __future__ import absolute_import, division, print_function

import os
import mmap
import warnings
from collections import OrderedDict

import numpy as np

from glue.core.data import Component, Data
from glue.config import data_factory


__all__ = ['is_hdf5', 'hdf5_reader']


def mmap_info_to_array(info, mapping):
    shape = info['shape']
    dtype = info['dtype']
    offset = info['offset']
    length = np.prod(shape)
    return np.frombuffer(mapping, dtype=dtype, count=length, offset=offset).reshape(shape)


def extract_hdf5_datasets(filename, memmap=True):
    '''
    Recursive function that returns a dictionary with all the datasets found in
    an HDF5 file or group. `handle` should be an instance of h5py.highlevel.File
    or h5py.highlevel.Group.
    '''

    import h5py

    file_handle = h5py.File(filename, 'r')

    arrays = {}

    from astropy.table import Table

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            full_path = item.name
            if item.dtype.kind in ('f', 'i', 'S'):
                offset = item.id.get_offset()
                # If an offset is available, the data is contiguous and we can
                # use memory mapping for efficiency.
                if not memmap or offset is None:
                    arrays[full_path] = item.value
                else:
                    arrays[full_path] = dict(offset=offset, shape=item.shape, dtype=item.dtype)
            elif item.dtype.kind in ('V',):
                arrays[full_path] = Table.read(item, format='hdf5')

    file_handle.visititems(visitor)
    file_handle.close()

    # Now create memory-mapped arrays

    with open(filename, 'rb') as data_file:
        fileno = data_file.fileno()
        mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
        for name, value in arrays.items():
            if isinstance(value, dict):
                arrays[name] = mmap_info_to_array(value, mapping)

    return arrays


def is_hdf5(filename):
    # All hdf5 files begin with the same sequence
    with open(filename, 'rb') as infile:
        return infile.read(8) == b'\x89HDF\r\n\x1a\n'


@data_factory(label="HDF5 file", identifier=is_hdf5, priority=100)
def hdf5_reader(filename, auto_merge=False, memmap=True, **kwargs):
    """
    Read in all datasets from an HDF5 file

    Parameters
    ----------
    filename : str
        The filename of the HDF5 file
    memmap : bool, optional
        Whether to use memory mapping
    """

    from astropy.table import Table

    # Read in all datasets
    datasets = extract_hdf5_datasets(filename, memmap=memmap)

    label_base = os.path.basename(filename).rpartition('.')[0]

    if not label_base:
        label_base = os.path.basename(filename)

    data_by_shape = {}

    groups = OrderedDict()

    for key in datasets:
        label = '{0}[{1}]'.format(label_base, key)
        array = datasets[key]
        if isinstance(array, Table):
            data = Data(label=label)
            groups[label] = data
            for column_name in array.columns:
                column = array[column_name]
                if column.ndim == 1:
                    component = Component.autotyped(column, units=column.unit)
                    data.add_component(component=component,
                                       label=column_name)
                else:
                    warnings.warn("HDF5: Ignoring vector column {0}".format(column_name))
        else:
            if auto_merge and array.shape in data_by_shape:
                data = data_by_shape[datasets[key].shape]
            else:
                data = Data(label=label)
                data_by_shape[array.shape] = data
                groups[label] = data
            data.add_component(component=datasets[key], label=key[1:])

    return [groups[idx] for idx in groups]
