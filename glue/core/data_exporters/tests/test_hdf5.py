import h5py
import numpy as np
from glue.core import Data

from ..hdf5 import hdf5_writer


def test_hdf5_writer_data(tmpdir):

    filename = tmpdir.join('test1.hdf5').strpath

    data = Data(x=np.arange(6).reshape(2, 3),
                y=(np.arange(6) * 2).reshape(2, 3))

    hdf5_writer(filename, data)

    f = h5py.File(filename)
    assert len(f) == 2
    np.testing.assert_equal(f['x'].value, data['x'])
    np.testing.assert_equal(f['y'].value, data['y'])
    f.close()

    # Only write out some components

    filename = tmpdir.join('test2.hdf5').strpath

    hdf5_writer(filename, data, components=[data.id['x']])

    f = h5py.File(filename)
    assert len(f) == 1
    np.testing.assert_equal(f['x'].value, data['x'])
    f.close()


def test_hdf5_writer_subset(tmpdir):

    filename = tmpdir.join('test').strpath

    data = Data(x=np.arange(6).reshape(2, 3).astype(float),
                y=(np.arange(6) * 2).reshape(2, 3).astype(float))

    subset = data.new_subset()
    subset.subset_state = data.id['x'] > 2

    hdf5_writer(filename, subset)

    f = h5py.File(filename)
    assert np.all(np.isnan(f['x'].value[0]))
    assert np.all(np.isnan(f['y'].value[0]))
    np.testing.assert_equal(f['x'].value[1], data['x'][1])
    np.testing.assert_equal(f['y'].value[1], data['y'][1])
    f.close()
