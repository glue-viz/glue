import os

import numpy as np
from glue.core import Data
from astropy.io import fits

from ..gridded_fits import fits_writer


def test_fits_writer(tmpdir):

    filename = tmpdir.join('test').strpath

    data = Data(x=np.arange(6).reshape(2, 3),
                y=(np.arange(6) * 2).reshape(2, 3))

    fits_writer(filename, data)

    with fits.open(filename) as hdulist:
        np.testing.assert_equal(hdulist['x'].data, data['x'])
        np.testing.assert_equal(hdulist['y'].data, data['y'])
