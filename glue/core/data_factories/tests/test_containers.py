from __future__ import absolute_import, division, print_function

import os
from collections import namedtuple

from ....tests.helpers import requires_astropy

from ... import data_factories as df


DATA = os.path.join(os.path.dirname(__file__), 'data')

Expected = namedtuple('Expected', 'shape, ndim')


@requires_astropy
def test_container_fits():
    expected = {
        'generic[ATAB]': Expected(
            shape=(20,),
            ndim=1
        ),
        'generic[TWOD]': Expected(
            shape=(4, 5),
            ndim=2
        ),
        'generic[ONED]': Expected(
            shape=(20,),
            ndim=1
        ),
        'generic[THREED]': Expected(
            shape=(2, 2, 5),
            ndim=3
        )
    }

    d_set = df.load_data(
        '/'.join([DATA, 'generic.fits']),
        factory=df.fits_container
    )
    assert len(d_set) == 4
    for d in d_set:
        e = expected[d.label]
        assert e.shape == d.shape
        assert e.ndim == d.ndim
