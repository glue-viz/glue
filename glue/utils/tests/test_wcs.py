import pytest
import numpy as np
from numpy.testing import assert_equal

# The functionality we are testing relies on the new WCS API (APE 14) in Astropy
astropy = pytest.importorskip("astropy", minversion="3.1")

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from glue.utils.wcs import (axis_correlation_matrix,
                            pixel_to_world_correlation_matrix,
                            pixel_to_pixel_correlation_matrix,
                            split_matrix)


def test_pixel_to_world_correlation_matrix_celestial():

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs.wcs.set()

    assert_equal(axis_correlation_matrix(wcs), [[1, 1], [1, 1]])
    matrix, classes = pixel_to_world_correlation_matrix(wcs)
    assert_equal(matrix, [[1, 1]])
    assert classes == [SkyCoord]


def test_pixel_to_world_correlation_matrix_spectral_cube_uncorrelated():

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = 'RA---TAN', 'FREQ', 'DEC--TAN'
    wcs.wcs.set()

    assert_equal(axis_correlation_matrix(wcs), [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    matrix, classes = pixel_to_world_correlation_matrix(wcs)
    assert_equal(matrix, [[1, 0, 1], [0, 1, 0]])
    assert classes == [SkyCoord, Quantity]


def test_pixel_to_world_correlation_matrix_spectral_cube_correlated():

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = 'RA---TAN', 'FREQ', 'DEC--TAN'
    wcs.wcs.cd = np.ones((3, 3))
    wcs.wcs.set()

    assert_equal(axis_correlation_matrix(wcs), [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    matrix, classes = pixel_to_world_correlation_matrix(wcs)
    assert_equal(matrix, [[1, 1, 1], [1, 1, 1]])
    assert classes == [SkyCoord, Quantity]


def test_pixel_to_pixel_correlation_matrix_celestial():

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs1.wcs.set()

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs2.wcs.set()

    matrix = pixel_to_pixel_correlation_matrix(wcs1, wcs2)
    assert_equal(matrix, [[1, 1], [1, 1]])


def test_pixel_to_pixel_correlation_matrix_spectral_cube_uncorrelated():

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'FREQ'
    wcs1.wcs.set()

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs2.wcs.set()

    matrix = pixel_to_pixel_correlation_matrix(wcs1, wcs2)
    assert_equal(matrix, [[1, 1, 0], [0, 0, 1], [1, 1, 0]])


def test_pixel_to_pixel_correlation_matrix_spectral_cube_correlated():

    # NOTE: only make one of the WCSes have correlated axes to really test this

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'FREQ'
    wcs1.wcs.set()

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs2.wcs.cd = np.ones((3, 3))
    wcs2.wcs.set()

    matrix = pixel_to_pixel_correlation_matrix(wcs1, wcs2)
    assert_equal(matrix, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def test_pixel_to_pixel_correlation_matrix_mismatch():

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs1.wcs.set()

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs2.wcs.set()

    with pytest.raises(ValueError) as exc:
        pixel_to_pixel_correlation_matrix(wcs1, wcs2)
    assert exc.value.args[0] == "The two WCS return a different number of world coordinates"

    wcs3 = WCS(naxis=2)
    wcs3.wcs.ctype = 'FREQ', 'PIXEL'
    wcs3.wcs.set()

    with pytest.raises(ValueError) as exc:
        pixel_to_pixel_correlation_matrix(wcs2, wcs3)
    assert exc.value.args[0] == "The world coordinate types of the two WCS don't match"

    wcs4 = WCS(naxis=4)
    wcs4.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'Q1', 'Q2'
    wcs4.wcs.cunit = ['deg', 'deg', 'm/s', 'm/s']
    wcs4.wcs.set()

    wcs5 = WCS(naxis=4)
    wcs5.wcs.ctype = 'Q1', 'RA---TAN', 'DEC--TAN', 'Q2'
    wcs5.wcs.cunit = ['m/s', 'deg', 'deg', 'm/s']
    wcs5.wcs.set()

    with pytest.raises(ValueError) as exc:
        pixel_to_pixel_correlation_matrix(wcs4, wcs5)
    assert exc.value.args[0] == "World coordinate order doesn't match and automatic matching is ambiguous"


def test_split_matrix():

    assert split_matrix(np.array([[1]])) == [([0], [0])]

    assert split_matrix(np.array([[1, 1],
                                  [1, 1]])) == [([0, 1], [0, 1])]

    assert split_matrix(np.array([[1, 1, 0],
                                  [1, 1, 0],
                                  [0, 0, 1]])) == [([0, 1], [0, 1]), ([2], [2])]

    assert split_matrix(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]])) == [([0], [1]), ([1], [0]), ([2], [2])]

    assert split_matrix(np.array([[0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 1]])) == [([0, 1, 2], [0, 1, 2])]
