import os

import pytest
import numpy as np
from astropy import wcs
from astropy.io import fits

from .. import pvregions


def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, filename)


@pytest.mark.xfail
def test_wcspath():
    p1,p2 = pvregions.paths_from_regfile(data_path('tests.reg'))
    w = wcs.WCS(fits.Header.fromtextfile(data_path('w51.hdr')))
    p1.set_wcs(w)

    shouldbe = (61.593585, 75.950471, 88.911616, 72.820463, 96.025686,
                88.470505, 113.95314, 95.868707, 186.80123, 76.235017,
                226.35546, 100.99054)
    xy = (np.array(zip(shouldbe[::2],shouldbe[1::2])) - 1)
    np.testing.assert_allclose(np.array((p1.xy)), xy, rtol=1e-5)
