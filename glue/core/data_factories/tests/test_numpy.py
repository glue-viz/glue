from __future__ import absolute_import, division, print_function

# Third-party
import numpy as np
from numpy.testing import assert_array_equal

# Package
from glue.core import data_factories as df


def test_npy_load(tmpdir):
    data = np.array([("a", 152.2352, -21.513), ("b", 21.412, 35.1341)],
                    dtype=[('name', '|S1'), ('ra', 'f8'), ('dec', 'f8')])

    with open(tmpdir.join('test.npy').strpath, 'wb') as f:
        np.save(f, data)
        f.seek(0)

        data2 = df.load_data(f.name)
        assert_array_equal(data['name'], data2.get_component('name').labels)
        assert_array_equal(data['ra'], data2['ra'])
        assert_array_equal(data['dec'], data2['dec'])
        assert data2.label == 'test'


def test_unstruc_npy_load(tmpdir):
    data = np.array([[152.2352, -21.513], [21.412, 35.1341]], dtype='f8')

    with open(tmpdir.join('test.npy').strpath, 'wb') as f:
        np.save(f, data)
        f.seek(0)

        data2 = df.load_data(f.name)
        assert_array_equal(data, data2['array'])
        assert data2.label == 'test'


def test_unstruc_npz_load(tmpdir):
    data1 = np.array([[152.2352, -21.513], [21.412, 35.1341]], dtype='f8')
    data2 = np.array([[15.2352, -2.513], [2.412, 3.1341]], dtype='f8')

    with open(tmpdir.join('test.npz').strpath, 'wb') as f:
        np.savez(f, data1=data1, data2=data2)
        f.seek(0)

        data_loaded = df.load_data(f.name)
        arr = data_loaded[0]
        assert arr.label == 'data1'
        assert_array_equal(data1, arr['array'])

        arr = data_loaded[1]
        assert arr.label == 'data2'
        assert_array_equal(data2, arr['array'])


def test_npz_load(tmpdir):
    data1 = np.array([("a", 152.2352, -21.513), ("b", 21.412, 35.1341)],
                     dtype=[('name', '|S1'), ('ra', 'f8'), ('dec', 'f8')])
    data2 = np.array([("c", 15.2352, -2.513), ("d", 2.412, 3.1341)],
                     dtype=[('name', '|S1'), ('l', 'f8'), ('b', 'f8')])

    with open(tmpdir.join('test.npz').strpath, 'wb') as f:

        np.savez(f, data1=data1, data2=data2)
        f.seek(0)

        data_loaded = df.load_data(f.name)

        arr = data_loaded[0]
        assert arr.label == 'data1'
        assert_array_equal(data1['name'], arr.get_component('name').labels)
        assert_array_equal(data1['ra'], arr['ra'])
        assert_array_equal(data1['dec'], arr['dec'])

        arr = data_loaded[1]
        assert arr.label == 'data2'
        assert_array_equal(data2['name'], arr.get_component('name').labels)
        assert_array_equal(data2['l'], arr['l'])
        assert_array_equal(data2['b'], arr['b'])
