from __future__ import absolute_import, division, print_function

import os
from collections import namedtuple
from copy import deepcopy

import numpy as np

from ....tests.helpers import requires_astropy

from ... import data_factories as df
from ..containers import fits_container

DATA = os.path.join(os.path.dirname(__file__), 'data')

Expected = namedtuple('Expected', 'shape, ndim')


def _assert_equal_expected(actual, expected):
    assert len(actual) == len(expected)
    for d in actual:
        e = expected[d.label]
        assert e.shape == d.shape
        assert e.ndim == d.ndim


@requires_astropy
def test_container_fits():

    from ....external.astro import fits

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

    # Make sure the factory gets used

    d_set = df.load_data(os.path.join(DATA, 'generic.fits'),
                         factory=df.fits_container)

    _assert_equal_expected(d_set, expected)

    # Check that fits_container takes HDUList objects

    hdulist = fits.open(os.path.join(DATA, 'generic.fits'))
    d_set = fits_container(hdulist)

    _assert_equal_expected(d_set, expected)

    # Sometimes the primary HDU is empty but with an empty array rather than
    # None

    hdulist[0].data = np.array([])
    d_set = fits_container(hdulist)

    _assert_equal_expected(d_set, expected)

    # Check that exclude_exts works

    d_set = fits_container(hdulist, exclude_exts=['TWOD'])
    expected_reduced = deepcopy(expected)
    expected_reduced.pop('generic[TWOD]')

    _assert_equal_expected(d_set, expected_reduced)


@requires_astropy
def test_auto_merge_fits():

    from ....external.astro import fits

    expected = {
        'HDUList[A]': Expected(
            shape=(3, 4),
            ndim=2
        ),
        'HDUList[B]': Expected(
            shape=(3, 4),
            ndim=2
            )
    }

    # Check that merging works

    data = np.ones((3,4))
    hdu1 = fits.ImageHDU(data)
    hdu1.name = 'a'
    hdu2 = fits.ImageHDU(data)
    hdu2.name = 'b'
    hdulist = fits.HDUList([hdu1, hdu2])

    d_set = fits_container(hdulist)

    _assert_equal_expected(d_set, expected)

    expected = {
        'HDUList[A]': Expected(
            shape=(3, 4),
            ndim=2
        ),
    }

    d_set = fits_container(hdulist, auto_merge=True)

    _assert_equal_expected(d_set, expected)

    d_set[0].get_component('A')
    d_set[0].get_component('B')
