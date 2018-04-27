from __future__ import absolute_import, division, print_function

import pytest

import numpy as np
from astropy.table import Table

from glue.core import Data
from ..astropy_table import (ipac_exporter, latex_exporter,
                             votable_exporter, fits_exporter)

EXPORTERS = {}
EXPORTERS['ascii.ipac'] = ipac_exporter
EXPORTERS['ascii.latex'] = latex_exporter
EXPORTERS['votable'] = votable_exporter
EXPORTERS['fits'] = fits_exporter


@pytest.mark.parametrize('fmt', EXPORTERS)
def test_astropy_table(tmpdir, fmt):

    filename = tmpdir.join('test1').strpath
    data = Data(x=[1, 2, 3], y=[b'a', b'b', b'c'])
    EXPORTERS[fmt](filename, data)
    t = Table.read(filename, format=fmt)
    assert t.colnames == ['x', 'y']
    np.testing.assert_equal(t['x'], [1, 2, 3])
    np.testing.assert_equal(t['y'].astype(bytes), [b'a', b'b', b'c'])

    filename = tmpdir.join('test2').strpath
    EXPORTERS[fmt](filename, data, components=[data.id['x']])
    t = Table.read(filename, format=fmt)
    assert t.colnames == ['x']
    np.testing.assert_equal(t['x'], [1, 2, 3])
