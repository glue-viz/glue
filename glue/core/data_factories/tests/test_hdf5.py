import numpy as np

from ..helpers import auto_data
from ....tests.helpers import requires_h5py


@requires_h5py
def test_skip_non_numerical(tmpdir):

    # This is a regression for a bug that caused the HDF5 loader to crash if it
    # encountered HDF5 datasets that were not numerical. For instance, a dataset
    # with one string element does not have a shape so caused a crash.

    filename = tmpdir.join('test.hdf5').strpath

    import h5py

    f = h5py.File(filename, 'w')
    f.create_dataset('a', data='hello')
    f.create_dataset('b', data=np.array([1,2,3]))
    f.close()

    auto_data(filename)
