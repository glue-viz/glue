# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import patch
from numpy.testing import assert_allclose

from glue.tests.helpers import requires_astropy

from ..coordinates import (coordinates_from_header,
                           WCSCoordinates,
                           Coordinates,
                           header_from_string)


@requires_astropy
class TestWcsCoordinates(object):

    def default_header(self):
        from astropy.io import fits
        hdr = fits.Header()
        hdr['NAXIS'] = 2
        hdr['CRVAL1'] = 0
        hdr['CRVAL2'] = 5
        hdr['CRPIX1'] = 250
        hdr['CRPIX2'] = 187.5
        hdr['CTYPE1'] = 'GLON-TAN'
        hdr['CTYPE2'] = 'GLAT-TAN'
        hdr['CD1_1'] = -0.0166666666667
        hdr['CD1_2'] = 0.
        hdr['CD2_1'] = 0.
        hdr['CD2_2'] = 0.01666666666667
        return hdr

    def test_pixel2world_scalar(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = 250., 187.5
        result = coord.pixel2world(x, y)
        expected = 359.9832692105993601, 5.0166664867400375
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_different_input_types(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = 250, 187.5
        result = coord.pixel2world(x, y)
        expected = 359.9832692105993601, 5.0166664867400375
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_list(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = [250, 250], [187.5, 187.5]
        result = coord.pixel2world(x, y)
        expected = ([359.9832692105993601, 359.9832692105993601],
                    [5.0166664867400375, 5.0166664867400375])

        for i in range(0, 1):
            for r, e in zip(result[i], expected[i]):
                assert_allclose(r, e)

    def test_pixel2world_numpy(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = np.array([250, 250]), np.array([187.5, 187.5])
        result = coord.pixel2world(x, y)
        expected = (np.array([359.9832692105993601, 359.9832692105993601]),
                    np.array([5.0166664867400375, 5.0166664867400375]))

        np.testing.assert_array_almost_equal(result[0], expected[0])
        np.testing.assert_array_almost_equal(result[1], expected[1])

    def test_world2pixel_numpy(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = np.array([0, 0]), np.array([0, 0])
        expected = (np.array([249.0000000000000284, 249.0000000000000284]),
                    np.array([-114.2632689899972434, -114.2632689899972434]))

        result = coord.world2pixel(x, y)
        np.testing.assert_array_almost_equal(result[0], expected[0], 3)
        np.testing.assert_array_almost_equal(result[1], expected[1], 3)

    def test_world2pixel_list(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = [0, 0], [0, 0]
        expected = ([249.0000000000000284, 249.0000000000000284],
                    [-114.2632689899972434, -114.2632689899972434])

        result = coord.world2pixel(x, y)
        for i in range(0, 1):
            for r, e in zip(result[i], expected[i]):
                assert_allclose(r, e)

    def test_world2pixel_scalar(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        expected = 249.0000000000000284, -114.2632689899972434
        x, y = 0, 0

        result = coord.world2pixel(x, y)
        assert_allclose(result[0], expected[0], 3)
        assert_allclose(result[1], expected[1], 3)

    def test_world2pixel_mismatched_input(self):
        coord = WCSCoordinates(self.default_header())
        x, y = 0., [0.]
        expected = coord.world2pixel(x, y[0])

        result = coord.world2pixel(x, y)
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_mismatched_input(self):
        coord = WCSCoordinates(self.default_header())
        x, y = [250.], 187.5
        expected = coord.pixel2world(x[0], y)

        result = coord.pixel2world(x, y)
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_invalid_input(self):
        coord = WCSCoordinates(None)
        x, y = {}, {}
        with pytest.raises(TypeError) as exc:
            coord.pixel2world(x, y)

    def test_world2pixel_invalid_input(self):
        coord = WCSCoordinates(None)
        x, y = {}, {}
        with pytest.raises(TypeError) as exc:
            coord.world2pixel(x, y)

    def test_axis_label(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        assert coord.axis_label(0) == 'Galactic Latitude'
        assert coord.axis_label(1) == 'Galactic Longitude'


@requires_astropy
def test_world_axis_wcs():

    from astropy.io import fits

    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['CRVAL1'] = 0
    hdr['CRVAL2'] = 5
    hdr['CRPIX1'] = 2
    hdr['CRPIX2'] = 1
    hdr['CTYPE1'] = 'XOFFSET'
    hdr['CTYPE2'] = 'YOFFSET'
    hdr['CD1_1'] = -2.
    hdr['CD1_2'] = 0.
    hdr['CD2_1'] = 0.
    hdr['CD2_2'] = 2.

    data = np.ones((3, 4))
    coord = WCSCoordinates(hdr)
    assert_allclose(coord.world_axis(data, 0), [5, 7, 9])
    assert_allclose(coord.world_axis(data, 1), [2, 0, -2, -4])


class TestCoordinatesFromHeader(object):

    def test_2d_nowcs(self):
        hdr = {"NAXIS": 2}
        coord = coordinates_from_header(hdr)
        assert type(coord) == Coordinates

    def test_2d(self):
        hdr = header_from_string(HDR_2D_VALID)
        coord = coordinates_from_header(hdr)
        assert type(coord) == WCSCoordinates

    def test_3d_nowcs(self):
        hdr = HDR_3D_VALID_NOWCS
        coord = coordinates_from_header(hdr)
        assert type(coord) == Coordinates

    def test_3d(self):
        hdr = header_from_string(HDR_3D_VALID_WCS)
        coord = coordinates_from_header(hdr)
        assert type(coord) == WCSCoordinates

    def test_nod(self):
        hdr = 0
        coord = coordinates_from_header(hdr)
        assert type(coord) == Coordinates

HDR_2D_VALID = """
SIMPLE  =                    T / Written by IDL:  Wed Jul 27 10:01:47 2011
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    2 / number of data axes
NAXIS1  =                  501 / length of data axis 1
NAXIS2  =                  376 / length of data axis 2
EXTEND  =                    T / FITS dataset may contain extensions
RADESYS = 'FK5     '           / Frame of reference
CRVAL1  =                   0. / World coordinate 1 at reference point
CRVAL2  =                   5. / World coordinate 2 at reference point
CRPIX1  =              250.000 / Pixel coordinate 1 at reference point
CRPIX2  =              187.500 / Pixel coordinate 2 at reference point
CTYPE1  = 'GLON-TAN'           / Projection type
CTYPE2  = 'GLAT-TAN'           / Projection type
CUNIT1  = 'deg     '           / Unit used for axis 1
CUNIT2  = 'deg     '           / Unit used for axis 2
CD1_1   =         -0.016666667 / Pixel trasformation matrix
CD1_2   =                   0.
CD2_1   =                   0.
CD2_2   =          0.016666667
"""

HDR_3D_VALID_NOWCS = """SIMPLE  = T / Written by IDL:  Fri Mar 18 11:58:30 2011
BITPIX  =                  -32 / Number of bits per data pixel
NAXIS   =                    3 / Number of data axes
NAXIS1  =                  128 /
NAXIS2  =                  128 /
NAXIS3  =                  128 /
"""

HDR_3D_VALID_WCS = """SIMPLE  =                    T / Written by IDL:  Thu Jul  7 15:37:21 2011
BITPIX  =                  -32 / Number of bits per data pixel
NAXIS   =                    3 / Number of data axes
NAXIS1  =                   82 /
NAXIS2  =                   82 /
NAXIS3  =                  248 /
DATE    = '2011-07-07'         / Creation UTC (CCCC-MM-DD) date of FITS header
COMMENT FITS (Flexible Image Transport System) format is defined in 'Astronomy
COMMENT and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H
CTYPE1  = 'RA---CAR'           /
CTYPE2  = 'DEC--CAR'           /
CTYPE3  = 'VELO-LSR'           /
CRVAL1  =              55.3500 /
CRPIX1  =              41.5000 /
CDELT1  =    -0.00638888900000 /
CRVAL2  =              31.8944 /
CRPIX2  =              41.5000 /
CDELT2  =     0.00638888900000 /
CRVAL3  =       -9960.07902777 /
CRPIX3  =             -102.000 /
CDELT3  =        66.4236100000 /
"""


@requires_astropy
def test_coords_preserve_shape_2d():
    coord = coordinates_from_header(header_from_string(HDR_2D_VALID))
    x = np.zeros(12)
    y = np.zeros(12)
    result = coord.pixel2world(x, y)
    for r in result:
        assert r.shape == x.shape
    result = coord.world2pixel(x, y)
    for r in result:
        assert r.shape == x.shape

    x.shape = (4, 3)
    y.shape = (4, 3)
    result = coord.pixel2world(x, y)
    for r in result:
        assert r.shape == x.shape
    result = coord.world2pixel(x, y)
    for r in result:
        assert r.shape == x.shape

    x.shape = (2, 2, 3)
    y.shape = (2, 2, 3)
    result = coord.pixel2world(x, y)
    for r in result:
        assert r.shape == x.shape
    result = coord.world2pixel(x, y)
    for r in result:
        assert r.shape == x.shape


@requires_astropy
def test_coords_preserve_shape_3d():
    coord = coordinates_from_header(header_from_string(HDR_3D_VALID_NOWCS))
    x = np.zeros(12)
    y = np.zeros(12)
    z = np.zeros(12)
    result = coord.pixel2world(x, y, z)
    for r in result:
        assert r.shape == x.shape
    result = coord.world2pixel(x, y, z)
    for r in result:
        assert r.shape == x.shape

    x.shape = (4, 3)
    y.shape = (4, 3)
    z.shape = (4, 3)
    result = coord.pixel2world(x, y, z)
    for r in result:
        assert r.shape == x.shape
    result = coord.world2pixel(x, y, z)
    for r in result:
        assert r.shape == x.shape

    x.shape = (2, 2, 3)
    y.shape = (2, 2, 3)
    z.shape = (2, 2, 3)
    result = coord.pixel2world(x, y, z)
    for r in result:
        assert r.shape == x.shape
    result = coord.world2pixel(x, y, z)
    for r in result:
        assert r.shape == x.shape

def test_world_axis_unit():
    coord = coordinates_from_header(header_from_string(HDR_3D_VALID_WCS))
    assert coord.world_axis_unit(0) == 'm / s'
    assert coord.world_axis_unit(1) == 'deg'
    assert coord.world_axis_unit(2) == 'deg'
