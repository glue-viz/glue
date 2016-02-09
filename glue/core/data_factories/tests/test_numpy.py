from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy
from collections import namedtuple
import tempfile

import numpy as np
from numpy.testing import assert_array_equal

from glue.core import data_factories as df

from glue.tests.helpers import make_file

def test_npy_load():
    data = np.array([("a",152.2352,-21.513), ("b",21.412,35.1341)],
                    dtype=[('name','|S1'),('ra','f8'),('dec','f8')])

    with tempfile.NamedTemporaryFile(suffix='.npy') as f:
        np.save(f, data)
        f.seek(0)

        data2 = df.load_data(f.name)
        for name in data.dtype.names:
            assert_array_equal(data[name], data2[name])
