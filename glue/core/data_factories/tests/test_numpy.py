from __future__ import absolute_import, division, print_function

# Standard library
import tempfile

# Third-party
import numpy as np
from numpy.testing import assert_array_equal

# Package
from glue.core import data_factories as df

def test_npy_load():
    data = np.array([("a",152.2352,-21.513), ("b",21.412,35.1341)],
                    dtype=[('name','|S1'),('ra','f8'),('dec','f8')])

    with tempfile.NamedTemporaryFile(suffix='.npy') as f:
        np.save(f, data)
        f.seek(0)

        data2 = df.load_data(f.name)
        for name in data.dtype.names:
            assert_array_equal(data[name], data2[name])

def test_npz_load():
    data1 = np.array([("a",152.2352,-21.513), ("b",21.412,35.1341)],
                     dtype=[('name','|S1'),('ra','f8'),('dec','f8')])
    data2 = np.array([("c",15.2352,-2.513), ("d",2.412,3.1341)],
                     dtype=[('name','|S1'),('l','f8'),('b','f8')])

    with tempfile.NamedTemporaryFile(suffix='.npz') as f:
        np.savez(f, data1=data1, data2=data2)
        f.seek(0)

        data_loaded = df.load_data(f.name)

        arr = data_loaded[0]
        for name in data1.dtype.names:
            assert_array_equal(data1[name], arr[name])

        arr = data_loaded[1]
        for name in data2.dtype.names:
            assert_array_equal(data2[name], arr[name])
