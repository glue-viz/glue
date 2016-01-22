from __future__ import absolute_import, division, print_function

import os
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from glue.core import Data
from glue.tests.helpers import requires_astropy

from ..export_d3po import make_data_file


@requires_astropy
def test_make_data_file():

    from astropy.table import Table

    # astropy.Table interface has changed across versions. Check
    # that we build a valid table
    d = Data(x=[1, 2, 3], y=[2, 3, 4], label='data')
    s = d.new_subset(label='test')
    s.subset_state = d.id['x'] > 1

    dir = mkdtemp()
    try:
        make_data_file(d, (s,), dir)
        t = Table.read(os.path.join(dir, 'data.csv'), format='ascii')
        np.testing.assert_array_equal(t['x'], [1, 2, 3])
        np.testing.assert_array_equal(t['y'], [2, 3, 4])
        np.testing.assert_array_equal(t['selection_0'], [0, 1, 1])
    finally:
        rmtree(dir, ignore_errors=True)
