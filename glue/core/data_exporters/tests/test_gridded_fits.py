import os

import numpy as np
from glue.core import Data
from astropy.io import fits

from ..gridded_fits import fits_writer


def test_fits_writer_data(tmpdir):

    filename = tmpdir.join('test').strpath

    data = Data(x=np.arange(6).reshape(2, 3),
                y=(np.arange(6) * 2).reshape(2, 3))

    fits_writer(filename, data)

    with fits.open(filename) as hdulist:
        np.testing.assert_equal(hdulist['x'].data, data['x'])
        np.testing.assert_equal(hdulist['y'].data, data['y'])


def test_fits_writer_subset(tmpdir):

    filename = tmpdir.join('test').strpath

    data = Data(x=np.arange(6).reshape(2, 3),
                y=(np.arange(6) * 2).reshape(2, 3))

    subset = data.new_subset()
    subset.subset_state = data.id['x'] > 2

    fits_writer(filename, subset)

    with fits.open(filename) as hdulist:
        assert np.all(np.isnan(hdulist['x'].data[0]))
        assert np.all(np.isnan(hdulist['y'].data[0]))
        np.testing.assert_equal(hdulist['x'].data[1], data['x'][1])
        np.testing.assert_equal(hdulist['y'].data[1], data['y'][1])
