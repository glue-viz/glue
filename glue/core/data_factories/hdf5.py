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
    """
    Recursive function that returns a dictionary with all the datasets found in
    an HDF5 file or group. `handle` should be an instance of h5py.highlevel.File
    or h5py.highlevel.Group.

    Parameters
    ----------
    filename : str or file-like
        The path or file handle to the HDF5 file
    memmap : bool, optional
        Whether to use memory mapping
    """

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
                    arrays[full_path] = item[()]
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
def hdf5_reader(filename, auto_merge=True, memmap=True, **kwargs):
    """
    Read in all datasets from an HDF5 file

    Parameters
    ----------
    filename : str or file-like
        The path or file handle to the HDF5 file
    auto_merge : bool
        If all datasets have the same shape, and are at the base of the file,
        assume they are a column-based table and merge them into a single dataset.
    memmap : bool, optional
        Whether to use memory mapping
    """

    from astropy.table import Table

    # Read in all datasets
    datasets = extract_hdf5_datasets(filename, memmap=memmap)

    label_base = os.path.basename(filename).rpartition('.')[0]

    if not label_base:
        label_base = os.path.basename(filename)

    if len(datasets) == 0:
        return

    if not auto_merge or len(datasets) == 1 or any([isinstance(data, Table) for data in datasets.values()]):
        merge_data = False
    else:
        reference_shape = list(datasets.values())[0].shape
        merge_data = all([data.shape == reference_shape and key.count('/') == 1 for key, data in datasets.items()])

    groups = OrderedDict()

    data = None

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
            if data is None and merge_data:
                data = Data(label=label_base)
                groups[label_base] = data
            elif not merge_data:
                data = Data(label=label)
                groups[label] = data
            data.add_component(component=datasets[key], label=key[1:])

    return [groups[idx] for idx in sorted(groups)]
