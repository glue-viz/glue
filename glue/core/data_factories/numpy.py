from __future__ import absolute_import, division, print_function

from glue.core.data import Data, Component
from glue.config import data_factory
from glue.core.data_factories.helpers import has_extension

__all__ = ['is_npy_npz', 'npy_npz_reader']


def is_npy_npz(filename):
    """
    The first bytes are x93NUMPY (for npy) or PKx03x04 (for npz)
    """
    # See: https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
    from numpy.lib.format import MAGIC_PREFIX
    MAGIC_PREFIX_NPZ = b'PK\x03\x04'  # first 4 bytes for a zipfile
    tester = has_extension('npz .npz')
    with open(filename, 'rb') as infile:
        prefix = infile.read(6)
    return prefix == MAGIC_PREFIX or (tester(filename) and prefix[:4] == MAGIC_PREFIX_NPZ)


@data_factory(label="Numpy save file", identifier=is_npy_npz, priority=100)
def npy_npz_reader(filename, format='auto', auto_merge=False, **kwargs):
    """
    Read in a Numpy structured array saved to a .npy or .npz file.

    Parameters
    ----------
    source: str
        The pathname to the Numpy save file.
    """

    import numpy as np
    data = np.load(filename)

    if isinstance(data, np.ndarray):
        data = {None: data}

    groups = []
    for groupname in sorted(data):

        d = Data(label=groupname)
        arr = data[groupname]

        if arr.dtype.names is None:
            comp = Component.autotyped(arr)
            d.add_component(comp, label='array')
        else:
            for name in arr.dtype.names:
                comp = Component.autotyped(arr[name])
                d.add_component(comp, label=name)

        groups.append(d)

    return groups
