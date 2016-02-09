from __future__ import absolute_import, division, print_function

import os
import warnings
from collections import OrderedDict

from glue.core.data import Data, Component
from glue.config import data_factory


__all__ = ['is_npy', 'npy_reader']


def is_npy(filename):
    """
    The first bytes are: x93NUMPY
    see: https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
    """
    with open(filename, 'rb') as infile:
        return infile.read(6) == b'\x93NUMPY'


@data_factory(label="Numpy save file", identifier=is_npy)
def npy_reader(filename, format='auto', auto_merge=False, **kwargs):
    """
    Read in a Numpy structured array saved to a .npy or .npz file.

    Parameters
    ----------
    source: str
        The pathname to the Numpy save file.
    """

    import numpy as np
    arr = np.load(filename)

    d = Data()
    for name in arr.dtype.names:
        comp = Component(arr[name])
        d.add_component(comp, label=name)

    return d
