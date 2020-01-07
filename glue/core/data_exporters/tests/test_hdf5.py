import pytest
import numpy as np

from glue.core import Data
from glue.tests.helpers import requires_h5py

from ..hdf5 import hdf5_writer

DTYPES = [np.int16, np.int32, np.int64, np.float32, np.float64]


@requires_h5py
@pytest.mark.parametrize('dtype', DTYPES)
def test_hdf5_writer_data(tmpdir, dtype):

    filename = tmpdir.join('test1.hdf5').strpath

    data = Data(x=np.arange(6).reshape(2, 3).astype(dtype),
                y=(np.arange(6) * 2).reshape(2, 3).astype(dtype))

    hdf5_writer(filename, data)

    from h5py import File

    f = File(filename)
    assert len(f) == 2
    np.testing.assert_equal(f['x'][()], data['x'])
    np.testing.assert_equal(f['y'][()], data['y'])
    assert f['x'][()].dtype == dtype
    assert f['y'][()].dtype == dtype
    f.close()

    # Only write out some components

    filename = tmpdir.join('test2.hdf5').strpath

    hdf5_writer(filename, data, components=[data.id['x']])

    f = File(filename)
    assert len(f) == 1
    np.testing.assert_equal(f['x'][()], data['x'])
    f.close()


@requires_h5py
@pytest.mark.parametrize('dtype', DTYPES)
def test_hdf5_writer_subset(tmpdir, dtype):

    filename = tmpdir.join('test').strpath

    data = Data(x=np.arange(6).reshape(2, 3).astype(dtype),
                y=(np.arange(6) * 2).reshape(2, 3).astype(dtype))

    subset = data.new_subset()
    subset.subset_state = data.id['x'] > 2

    hdf5_writer(filename, subset)

    from h5py import File

    f = File(filename)

    if np.dtype(dtype).kind == 'f':
        assert np.all(np.isnan(f['x'][0]))
        assert np.all(np.isnan(f['y'][0]))
    else:
        np.testing.assert_equal(f['x'][0], 0)
        np.testing.assert_equal(f['y'][0], 0)

    np.testing.assert_equal(f['x'][1], data['x'][1])
    np.testing.assert_equal(f['y'][1], data['y'][1])
    assert f['x'][()].dtype == dtype
    assert f['y'][()].dtype == dtype
    f.close()
