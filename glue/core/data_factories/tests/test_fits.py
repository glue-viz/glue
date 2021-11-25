import os
from collections import namedtuple
from copy import deepcopy

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from glue.core import data_factories as df

from glue.tests.helpers import requires_astropy, make_file

from ..fits import fits_reader


DATA = os.path.join(os.path.dirname(__file__), 'data')

Expected = namedtuple('Expected', 'shape, ndim')


def _assert_equal_expected(actual, expected):
    assert len(actual) == len(expected)
    for d in actual:
        e = expected[d.label]
        assert e.shape == d.shape
        assert e.ndim == d.ndim


@requires_astropy
def test_component_unit():
    from astropy import units as u
    d_set = fits_reader(os.path.join(DATA, 'bunit.fits'))

    data = d_set[0]

    unit = u.Unit(data.get_component("ONED").units)
    assert unit == u.Jy


@requires_astropy
def test_container_fits():

    from astropy.io import fits

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
                         factory=df.fits_reader)

    _assert_equal_expected(d_set, expected)

    # Check that fits_reader takes HDUList objects

    with fits.open(os.path.join(DATA, 'generic.fits')) as hdulist:

        d_set = fits_reader(hdulist)

        _assert_equal_expected(d_set, expected)

        # Sometimes the primary HDU is empty but with an empty array rather than
        # None

        hdulist[0].data = np.array([])
        d_set = fits_reader(hdulist)

        _assert_equal_expected(d_set, expected)

        # Check that exclude_exts works

        d_set = fits_reader(hdulist, exclude_exts=['TWOD'])
        expected_reduced = deepcopy(expected)
        expected_reduced.pop('generic[TWOD]')

        _assert_equal_expected(d_set, expected_reduced)


@requires_astropy
def test_auto_merge_fits():

    from astropy.io import fits

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

    data = np.ones((3, 4))
    hdu1 = fits.ImageHDU(data)
    hdu1.name = 'a'
    hdu2 = fits.ImageHDU(data)
    hdu2.name = 'b'
    hdulist = fits.HDUList([hdu1, hdu2])

    d_set = fits_reader(hdulist)

    _assert_equal_expected(d_set, expected)

    expected = {
        'HDUList[A]': Expected(
            shape=(3, 4),
            ndim=2
        ),
    }

    d_set = fits_reader(hdulist, auto_merge=True)

    _assert_equal_expected(d_set, expected)

    d_set[0].get_component('A')
    d_set[0].get_component('B')


@requires_astropy
def test_fits_gz_factory():
    data = b'\x1f\x8b\x08\x08\xdd\x1a}R\x00\x03test.fits\x00\xed\xd1\xb1\n\xc20\x10\xc6q\x1f\xe5\xde@ZA]\x1cZ\x8d\x10\xd0ZL\x87\xe2\x16m\x0b\x1d\x9aHR\x87n>\xba\xa5".\tRq\x11\xbe_\xe6\xfb\x93\xe3\x04\xdf\xa7;F\xb4"\x87\x8c\xa6t\xd1\xaa\xd2\xa6\xb1\xd4j\xda\xf2L\x90m\xa5*\xa4)\\\x03D1\xcfR\x9e\xbb{\xc1\xbc\xefIcdG\x85l%\xb5\xdd\xb5tW\xde\x92(\xe7\x82<\xff\x0b\xfb\x9e\xba5\xe7\xd2\x90\xae^\xe5\xba)\x95\xad\xb5\xb2\xfe^\xe0\xed\x8d6\xf4\xc2\xdf\xf5X\x9e\xb1d\xe3\xbd\xc7h\xb1XG\xde\xfb\x06_\xf4N\xecx Go\x16.\xe6\xcb\xf1\xbdaY\x00\x00\x00\x80?r\x9f<\x1f\x00\x00\x00\x00\x00|\xf6\x00\x03v\xd8\xf6\x80\x16\x00\x00'

    with make_file(data, '.fits.gz') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.fits_reader

    assert_array_equal(d['PRIMARY'], [[0, 0], [0, 0]])


@requires_astropy
def test_casalike():
    d = df.load_data(os.path.join(DATA, 'casalike.fits'), factory=df.casalike_cube)
    assert d.shape == (1, 2, 2, 2)
    d['STOKES 0']
    d['STOKES 1']
    d['STOKES 2']
    d['STOKES 3']


# zlib-compressed
TEST_FITS_DATA = b'x\x9c\xed\xd0\xb1\n\xc20\x14\x85\xe1\xaa/r\xde@\x8a\xe2\xe6\xa0X!\xa0\xa5\xd0\x0c]\xa3m\xa1C\x13I\xe2\xd0\xb7\xb7b\xc5\xa1)\xe2\xe6p\xbe\xe5N\xf7\xe7rsq\xceN\t\xb0E\x80\xc4\x12W\xa3kc[\x07op\x142\x87\xf3J\x97\xca\x96\xa1\x05`/d&\x8apo\xb3\xee{\xcaZ\xd5\xa1T^\xc1w\xb7*\\\xf9Hw\x85\xc81q_\xdc\xf7\xf4\xbd\xbdT\x16\xa6~\x97\x9b\xb6\xd2\xae1\xdaM\xf7\xe2\x89\xde\xea\xdb5cI!\x93\xf40\xf9\xbf\xdf{\xcf\x18\x11\x11\x11\x11\xfd\xad\xe8e6\xcc\xf90\x17\x11\x11\x11\x11\x11\x11\x8d<\x00\x7fy\x7f\xbc'


@requires_astropy
def test_fits_image_loader():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.fits_reader
    assert_array_equal(d['PRIMARY'], [1, 2, 3])


@requires_astropy
def test_fits_uses_mmapping():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.fits_reader
    assert not d['PRIMARY'].flags['OWNDATA']


@requires_astropy
def test_fits_catalog_factory():
    data = b'\x1f\x8b\x08\x08\x19\r\x9cQ\x02\x03test.fits\x00\xed\xd7AO\x830\x18\xc6\xf1\xe9\'yo\x1c\'\x1c\x8c\x97\x1d\x86c\xa6\x911"5\xc1c\x91n\x92\x8cBJ\x97\xb8o\xef\x06\xd3\x98H\xdd\x16\x97]|~\x17\x12H\xfeyI{h\x136\x8b\xc3\x80hD=8\r\xe9\xb5R\x8bJ\x97\r\x99\x8a\xa6\x8c\'\xd4\x18\xa1r\xa1s\xea\xe53\x1e\xb3\xd4\xd2\xbb\xdb\xf6\x84\xd6bC\xb90\x82\xcc\xa6\x96t@4NYB\x96\xde\xcd\xb6\xa7\xd6e&5U\x8b\xcfrQJ\xd5\x14\x95jz{A\xca\x83hb\xfd\xdf\x93\xb51\x00\x00\x00\x00\xf87v\xc7\xc9\x84\xcd\xa3\x119>\x8b\xf8\xd8\x0f\x03\xe7\xdb\xe7!e\x85\x12zCFd+I\xf2\xddt\x87Sk\xef\xa2\xe7g\xef\xf4\xf3s\xdbs\xfb{\xee\xed\xb6\xb7\x92ji\xdev\xbd\xaf\x12\xb9\x07\xe6\xf3,\xf3\xb9\x96\x9eg\xef\xc5\xf7\xf3\xe7\x88\x1fu_X\xeaj]S-\xb4(\xa5\x91\xba\xff\x7f\x1f~\xeb\xb9?{\xcd\x81\xf5\xe0S\x16\x84\x93\xe4\x98\xf5\xe8\xb6\xcc\xa2\x90\xab\xdc^\xe5\xfc%\x0e\xda\xf5p\xc4\xfe\x95\xf3\x97\xfd\xcc\xa7\xf3\xa7Y\xd7{<Ko7_\xbb\xbeNv\xb6\xf9\xbc\xf3\xcd\x87\xfb\x1b\x00\x00\xc0\xe5\r:W\xfb\xe7\xf5\x00\x00\x00\x00\x00\x00\xac>\x00\x04\x01*\xc7\xc0!\x00\x00'
    with make_file(data, '.fits') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.fits_reader

    assert_array_equal(d['a'], [1])
    assert_array_equal(d['b'], [2])


@requires_astropy
def test_fits_compressed():
    # Regression test for bug that caused images with compressed image HDUs
    # to not be read
    d = df.load_data(os.path.join(DATA, 'compressed_image.fits'),
                     factory=df.fits_reader)
    assert d.ndim == 2


@requires_astropy
def test_fits_vector():
    # Regression test for bug that caused tables with vector columns to not load
    with pytest.warns(UserWarning, match="Dropping column 'status' since it is not 1-dimensional"):
        df.load_data(os.path.join(DATA, 'events.fits'), factory=df.fits_reader)


@requires_astropy
def test_save_meta():
    # Regression test for a bug that causes Data.meta to contain non-string
    # items when FITS comments were present.
    from glue.core.tests.test_state import clone
    data = df.load_data(os.path.join(DATA, 'comment.fits'))
    clone(data)


@requires_astropy
def test_coordinate_component_units():
    # Regression test for a bug that caused units to be missing from coordinate
    # components.
    d = df.load_data(os.path.join(DATA, 'casalike.fits'), factory=df.casalike_cube)
    wid = d.world_component_ids
    assert wid[0].label == 'Stokes'
    assert d.get_component(wid[0]).units == ''
    assert wid[1].label == 'Vopt'
    # astropy 5.0 changed the string representation of WCS units
    assert d.get_component(wid[1]).units in ['m s-1', 'm.s**-1']
    assert wid[2].label == 'Declination'
    assert d.get_component(wid[2]).units == 'deg'
    assert wid[3].label == 'Right Ascension'
    assert d.get_component(wid[3]).units == 'deg'
