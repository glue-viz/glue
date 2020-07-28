# 3rd-party
from astropy.nddata import NDData, StdDevUncertainty
from astropy.wcs import WCS

import numpy as np
from numpy.testing import assert_array_equal

# Package
from glue.core.data_factories.nddata import nddata_data_loader


def test_nddata_data_loader():
    data = np.array([1, 2, 3, 4])
    w = WCS(naxis=3)
    w.wcs.ctype = 'DEC--TAN', 'RA---TAN', 'WAVE'
    w.wcs.set()
    meta = {'test': 'meta_test', 'meta': 'test_meta'}
    mask = data > 2
    uncertainty = StdDevUncertainty(np.sqrt(data))
    unit = 'm / s'
    ndd = NDData(data=data, wcs=w, meta=meta, mask=mask, unit=unit, uncertainty=uncertainty)
    result = nddata_data_loader(ndd)

    assert_array_equal(ndd.data, result['data'])
    assert_array_equal(ndd.wcs, result.coords)
    assert_array_equal(ndd.meta, result.meta)
    assert_array_equal(ndd.mask, result.mask)
    assert_array_equal(ndd.uncertainty, result.uncertainty)
    assert_array_equal(ndd.unit, result.get_component('data').units)