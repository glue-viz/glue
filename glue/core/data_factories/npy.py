from __future__ import absolute_import, division, print_function

from glue.core.data import Data, Component
from glue.config import data_factory
from glue.core.data_factories.helpers import has_extension

__all__ = ['is_npy', 'npy_reader', 'is_npz', 'npz_reader', 'from_unstructured_array']

# TODO: implement support for regular arrays, e.g., not just structured arrays?

def is_npy(filename):
    """
    The first bytes are: x93NUMPY
    see: https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
    """
    from numpy.lib.format import MAGIC_PREFIX
    with open(filename, 'rb') as infile:
        return infile.read(6) == MAGIC_PREFIX

@data_factory(label="Numpy save file", identifier=is_npy, priority=100)
def npy_reader(filename, format='auto', auto_merge=False, **kwargs):
    """
    Read in a Numpy structured array saved to a .npy or .npz file.

    Parameters
    ----------
    source: str
        The pathname to the Numpy save file.
    """

    import numpy as np
    npy_data = np.load(filename)

    if isinstance(npy_data.dtype.names, type(None)):
        npy_data = from_unstructured_array(npy_data)

    d = Data()
    for name in npy_data.dtype.names:
        comp = Component.autotyped(npy_data[name])
        d.add_component(comp, label=name)

    return d

def from_unstructured_array(arr):
    """
    Copy an unstructured (and therefore of a single dtype) numpy array to a structured array
    of the same shape.

    Parameters
    ----------
    arr: numpy.ndarray
    """
    import numpy as np

    # creates an zero filled array with same shape and dtype as input array, field name is set to 'array'
    unstructured_array = np.core.records.recarray(arr.shape, names=['array'], formats=arr.dtype) 
    unstructured_array['array'] = arr

    return unstructured_array

def is_npz(filename):
    """
    The first bytes are: x93NUMPY
    see: https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
    """
    tester = has_extension('npz .npz')

    MAGIC_PREFIX = b'PK\x03\x04' # first 4 bytes for a zipfile
    with open(filename, 'rb') as infile:
        check = infile.read(4) == MAGIC_PREFIX
    return check and tester(filename)

@data_factory(label="Numpy multiple array save file", identifier=is_npz, priority=100)
def npz_reader(filename, format='auto', auto_merge=False, **kwargs):
    """
    Read in a Numpy structured array saved to a .npy or .npz file.

    Parameters
    ----------
    source: str
        The pathname to the Numpy save file.
    """

    import numpy as np
    npy_data = np.load(filename)

    groups = []
    for groupname in sorted(npy_data.files):
        d = Data(label=groupname)
        arr = npy_data[groupname]

        if isinstance(arr.dtype.names, type(None)):
            arr = from_unstructured_array(arr)

        for name in arr.dtype.names:
            comp = Component.autotyped(arr[name])
            d.add_component(comp, label=name)

        groups.append(d)

    return groups


