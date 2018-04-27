from __future__ import absolute_import, division, print_function

import numpy as np
from glue.core import Data
from astropy.io import fits

from ..gridded_fits import fits_writer


def test_fits_writer_data(tmpdir):

    filename = tmpdir.join('test1.fits').strpath

    data = Data(x=np.arange(6).reshape(2, 3),
                y=(np.arange(6) * 2).reshape(2, 3))

    fits_writer(filename, data)

    with fits.open(filename) as hdulist:
        assert len(hdulist) == 2
        np.testing.assert_equal(hdulist['x'].data, data['x'])
        np.testing.assert_equal(hdulist['y'].data, data['y'])

    # Only write out some components

    filename = tmpdir.join('test2.fits').strpath

    fits_writer(filename, data, components=[data.id['x']])

    with fits.open(filename) as hdulist:
        assert len(hdulist) == 1
        np.testing.assert_equal(hdulist['x'].data, data['x'])


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
