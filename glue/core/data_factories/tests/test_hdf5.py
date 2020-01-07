import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from glue.core import data_factories as df
from glue.tests.helpers import requires_h5py, requires_astropy, make_file

from ..helpers import auto_data

DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_h5py
def test_skip_non_numerical(tmpdir):

    # This is a regression for a bug that caused the HDF5 loader to crash if it
    # encountered HDF5 datasets that were not numerical. For instance, a dataset
    # with one string element does not have a shape so caused a crash.

    filename = tmpdir.join('test.hdf5').strpath

    import h5py

    f = h5py.File(filename, 'w')
    f.create_dataset('a', data='hello')
    f.create_dataset('b', data=np.array([1, 2, 3]))
    f.close()

    auto_data(filename)


@requires_astropy
@requires_h5py
@pytest.mark.parametrize('suffix', ['.h5', '.hdf5', '.hd5', '.h5custom'])
def test_hdf5_loader(suffix):
    data = b'x\xda\xeb\xf4pq\xe3\xe5\x92\xe2b\x00\x01\x0e\x0e\x06\x16\x06\x01\x06d\xf0\x1f\n*8P\xf90\xf9\x04(\xcd\x08\xa5;\xa0\xf4\n&\x988#XN\x02*.\x085\x1f]]H\x90\xab+H\xf5\x7f4\x00\xb3\xc7\x80\x05Bs0\x8c\x82\x91\x08<\\\x1d\x03@t\x04\x94\x0fK\xa5\'\x98P\xd5U\xa0\xa5G\x0f\n\xeded`\x83\x98\xc5\x08\xe3CR2##D\x80\x19\xaa\x0eA\x0b\x80\x95\np\xc0\xd2\xaa\x03\x98d\x05\xf2@\xe2LLL\x8c\x90t,\x01\xe633&@\x93\xb4\x04\x8a\xbdBP\xdd 5\xc9\xd5]A\x0c\x0c\r\x83"\x1e\x82\xfd\xfc]@9\x1a\x96\x0f\x15\x98G\xd3\xe6(\x18\x05\xa3\x00W\xf9\t\x01Lh\xe5$\x00\xc2A.\xaf'
    with make_file(data, suffix, decompress=True) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.hdf5_reader
    assert_array_equal(d['x'], [1, 2, 3])


@requires_astropy
@requires_h5py
def test_hdf5_loader_fromfile():

    datasets = df.load_data(os.path.join(DATA, 'data.hdf5'))

    if datasets[1].label < datasets[0].label:
        datasets = datasets[::-1]

    assert datasets[0].label == 'data[/a/tab]'
    assert_array_equal(datasets[0]['e'], [3, 2, 1])
    assert_array_equal(datasets[0]['f'], [1.5, 2.5, 1.0])
    assert_array_equal(datasets[0]['g'].categories, [b'a', b'b', b'c'])

    assert datasets[1].label == 'data[/x]'
    assert_array_equal(datasets[1]['x'], [1, 2, 3])


@requires_astropy
@requires_h5py
def test_hdf5_auto_merge(tmpdir):

    filename1 = tmpdir.mkdir('test1').join('test.hdf5').strpath
    filename2 = tmpdir.mkdir('test2').join('test.hdf5').strpath

    import h5py

    # When heterogeneous arrays are present, auto_merge is ignored

    f = h5py.File(filename1, 'w')
    f.create_dataset('a', data=np.array([[1, 2], [3, 4]]))
    f.create_dataset('b', data=np.array([1, 2, 3]))
    f.close()

    for auto_merge in [False, True]:

        datasets = df.hdf5_reader(filename1, auto_merge=auto_merge)

        assert len(datasets) == 2

        assert datasets[0].label == 'test[/a]'
        assert_array_equal(datasets[0]['a'], [[1, 2], [3, 4]])

        assert datasets[1].label == 'test[/b]'
        assert_array_equal(datasets[1]['b'], [1, 2, 3])

    # Check that the default is to merge if all arrays are homogeneous

    f = h5py.File(filename2, 'w')
    f.create_dataset('a', data=np.array([3, 4, 5]))
    f.create_dataset('b', data=np.array([1, 2, 3]))
    f.close()

    datasets = df.hdf5_reader(filename2)

    assert len(datasets) == 1

    assert datasets[0].label == 'test'
    assert_array_equal(datasets[0]['a'], [3, 4, 5])
    assert_array_equal(datasets[0]['b'], [1, 2, 3])

    # And check opt-out

    datasets = df.hdf5_reader(filename2, auto_merge=False)

    assert len(datasets) == 2

    assert datasets[0].label == 'test[/a]'
    assert_array_equal(datasets[0]['a'], [3, 4, 5])

    assert datasets[1].label == 'test[/b]'
    assert_array_equal(datasets[1]['b'], [1, 2, 3])
