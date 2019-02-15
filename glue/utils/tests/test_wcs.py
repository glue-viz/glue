import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

# The functionality we are testing relies on the new WCS API (APE 14) in Astropy
astropy = pytest.importorskip("astropy", minversion="3.1")

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from glue.utils import unbroadcast
from glue.utils.wcs import (axis_correlation_matrix,
                            pixel_to_world_correlation_matrix,
                            pixel_to_pixel_correlation_matrix,
                            split_matrix, efficient_pixel_to_pixel)


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

    # We need the following fix when celestial axes are correlated with
    # non-celestial axes: https://github.com/astropy/astropy/pull/8420
    pytest.importorskip("astropy", minversion="3.1.2")

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

    # We need the following fix when celestial axes are correlated with
    # non-celestial axes: https://github.com/astropy/astropy/pull/8420
    pytest.importorskip("astropy", minversion="3.1.2")

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


def test_pixel_to_pixel_correlation_matrix_nonsquare():

    # Here we set up an input WCS that maps 3 pixel coordinates to 4 world
    # coordinates - the idea is to make sure that things work fine in cases
    # where the number of input and output pixel coordinates don't match.

    class FakeWCS(object):
        pass

    wcs1 = FakeWCS()
    wcs1.pixel_n_dim = 3
    wcs1.world_n_dim = 4
    wcs1.axis_correlation_matrix = [[True, True, False],
                                    [True, True, False],
                                    [True, True, False],
                                    [False, False, True]]
    wcs1.world_axis_object_components = [('spat', 'ra', 'ra.degree'),
                                         ('spat', 'dec', 'dec.degree'),
                                         ('spec', 0, 'value'),
                                         ('time', 0, 'utc.value')]
    wcs1.world_axis_object_classes = {'spat': ('astropy.coordinates.SkyCoord', (),
                                               {'frame': 'icrs'}),
                                      'spec': ('astropy.units.Wavelength', (None,), {}),
                                      'time': ('astropy.time.Time', (None,),
                                               {'format': 'mjd', 'scale': 'utc'})}

    wcs2 = FakeWCS()
    wcs2.pixel_n_dim = 4
    wcs2.world_n_dim = 4
    wcs2.axis_correlation_matrix = [[True, False, False, False],
                                    [False, True, True, False],
                                    [False, True, True, False],
                                    [False, False, False, True]]
    wcs2.world_axis_object_components = [('spec', 0, 'value'),
                                         ('spat', 'ra', 'ra.degree'),
                                         ('spat', 'dec', 'dec.degree'),
                                         ('time', 0, 'utc.value')]
    wcs2.world_axis_object_classes = wcs1.world_axis_object_classes

    matrix = pixel_to_pixel_correlation_matrix(wcs1, wcs2)

    matrix = matrix.astype(int)

    # The shape should be (n_pixel_out, n_pixel_in)
    assert matrix.shape == (4, 3)

    expected = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 0, 1]])
    assert_equal(matrix, expected)


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


def test_efficient_pixel_to_pixel():

    wcs1 = WCS(naxis=3)
    wcs1.wcs.ctype = 'DEC--TAN', 'FREQ', 'RA---TAN'
    wcs1.wcs.set()

    wcs2 = WCS(naxis=3)
    wcs2.wcs.ctype = 'GLON-CAR', 'GLAT-CAR', 'FREQ'
    wcs2.wcs.set()

    # First try with scalars
    x, y, z = efficient_pixel_to_pixel(wcs1, wcs2, 1, 2, 3)
    assert x.shape == ()
    assert y.shape == ()
    assert z.shape == ()

    # Now try with broadcasted arrays
    x = np.linspace(10, 20, 10)
    y = np.linspace(10, 20, 20)
    z = np.linspace(10, 20, 30)
    Z1, Y1, X1 = np.meshgrid(z, y, x, indexing='ij', copy=False)
    X2, Y2, Z2 = efficient_pixel_to_pixel(wcs1, wcs2, X1, Y1, Z1)

    # The final arrays should have the correct shape
    assert X2.shape == (30, 20, 10)
    assert Y2.shape == (30, 20, 10)
    assert Z2.shape == (30, 20, 10)

    # But behind the scenes should also be broadcasted
    assert unbroadcast(X2).shape == (30, 1, 10)
    assert unbroadcast(Y2).shape == (30, 1, 10)
    assert unbroadcast(Z2).shape == (1, 20, 1)

    # We can put the values back through the function to ensure round-tripping
    X3, Y3, Z3 = efficient_pixel_to_pixel(wcs2, wcs1, X2, Y2, Z2)

    # The final arrays should have the correct shape
    assert X2.shape == (30, 20, 10)
    assert Y2.shape == (30, 20, 10)
    assert Z2.shape == (30, 20, 10)

    # But behind the scenes should also be broadcasted
    assert unbroadcast(X3).shape == (30, 1, 10)
    assert unbroadcast(Y3).shape == (1, 20, 1)
    assert unbroadcast(Z3).shape == (30, 1, 10)

    # And these arrays should match the input
    assert_allclose(X1, X3)
    assert_allclose(Y1, Y3)
    assert_allclose(Z1, Z3)


def test_efficient_pixel_to_pixel_simple():

    # Simple test to make sure that when WCS only returns one world coordinate
    # this still works correctly (since this requires special treatment behind
    # the scenes).

    wcs1 = WCS(naxis=2)
    wcs1.wcs.ctype = 'DEC--TAN', 'RA---TAN'
    wcs1.wcs.set()

    wcs2 = WCS(naxis=2)
    wcs2.wcs.ctype = 'GLON-CAR', 'GLAT-CAR'
    wcs2.wcs.set()

    # First try with scalars
    x, y = efficient_pixel_to_pixel(wcs1, wcs2, 1, 2)
    assert x.shape == ()
    assert y.shape == ()

    # Now try with broadcasted arrays
    x = np.linspace(10, 20, 10)
    y = np.linspace(10, 20, 20)
    Y1, X1 = np.meshgrid(y, x, indexing='ij', copy=False)
    Y2, X2 = efficient_pixel_to_pixel(wcs1, wcs2, X1, Y1)

    # The final arrays should have the correct shape
    assert X2.shape == (20, 10)
    assert Y2.shape == (20, 10)

    # and there are no efficiency gains here since the celestial axes are correlated
    assert unbroadcast(X2).shape == (20, 10)
