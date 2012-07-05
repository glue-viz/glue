import pytest

from pyfits import Header, Card
from mock import patch
import numpy as np
from numpy.testing import assert_almost_equal

from ..coordinates import coordinates_from_header, WCSCoordinates, Coordinates


class TestWcsCoordinates(object):

    def header_from_string(self, string):
        cards = []
        for s in string.splitlines():
            try:
                l, r = s.split('=')
                key = l.strip()
                value = r.split('/')[0].strip()
                try:
                    value = int(value)
                except ValueError:
                    pass
            except ValueError:
                continue
            cards.append(Card(key, value))
        return Header(cards)

    def default_header(self):
        hdr = Header()
        hdr.update('NAXIS', 2)
        hdr.update('CRVAL1', 0)
        hdr.update('CRVAL2', 5)
        hdr.update('CRPIX1', 250)
        hdr.update('CRPIX2', 187.5)
        hdr.update('CTYPE1', 'GLON-TAN')
        hdr.update('CTYPE2', 'GLAT-TAN')
        hdr.update('CD1_1', -0.0166666666667)
        hdr.update('CD1_2', 0.)
        hdr.update('CD2_1', 0.)
        hdr.update('CD2_2', 0.01666666666667)
        return hdr

    def test_pixel2world_scalar(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = 250., 187.5
        result = coord.pixel2world(x, y)
        expected = 0, 5
        assert_almost_equal(result[0], expected[0])
        assert_almost_equal(result[1], expected[1])

    def test_pixel2world_different_input_types(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = 250, 187.5
        result = coord.pixel2world(x, y)
        expected = 0, 5
        assert_almost_equal(result[0], expected[0])
        assert_almost_equal(result[1], expected[1])

    def test_pixel2world_list(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = [250, 250], [187.5, 187.5]
        result = coord.pixel2world(x, y)
        expected = [0, 0], [5, 5]
        for i in range(0,1):
            for r,e in zip(result[i], expected[i]):
                assert_almost_equal(r, e)

    def test_pixel2world_numpy(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = np.array([250, 250]), np.array([187.5, 187.5])
        result = coord.pixel2world(x, y)
        expected = np.array([0, 0]), np.array([5, 5])

        np.testing.assert_array_almost_equal(result[0], expected[0])
        np.testing.assert_array_almost_equal(result[1], expected[1])

    def test_world2pixel_numpy(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        expected = np.array([250, 250]), np.array([187.5, 187.5])
        x, y = np.array([0, 0]), np.array([5, 5])
        result = coord.world2pixel(x, y)

        np.testing.assert_array_almost_equal(result[0], expected[0])
        np.testing.assert_array_almost_equal(result[1], expected[1])

    def test_world2pixel_list(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        expected = [250, 250], [187.5, 187.5]
        x, y = [0, 0], [5, 5]
        result = coord.world2pixel(x, y)
        for i in range(0,1):
            for r,e in zip(result[i], expected[i]):
                assert_almost_equal(r, e)

    def test_world2pixel_scalar(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        expected = 250., 187.5
        x, y = 0, 5

        result = coord.world2pixel(x, y)
        assert_almost_equal(result[0], expected[0])
        assert_almost_equal(result[1], expected[1])

    def test_world2pixel_different_input_types(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        expected = 250., 187.5
        x, y = 0, 5.

        result = coord.world2pixel(x, y)
        assert_almost_equal(result[0], expected[0])
        assert_almost_equal(result[1], expected[1])

    def test_pixel2world_mismatched_input(self):
        coord = WCSCoordinates(None)
        x, y = 0, [5]
        with pytest.raises(TypeError):
            coord.pixel2world(x, y)

    def test_world2pixel_mismatched_input(self):
        coord = WCSCoordinates(None)
        x, y = 0, [5]
        with pytest.raises(TypeError):
            coord.world2pixel(x, y)

    def test_pixel2world_invalid_input(self):
        coord = WCSCoordinates(None)
        x, y = {}, {}
        with pytest.raises(TypeError):
            coord.pixel2world(x, y)

    def test_world2pixel_invalid_input(self):
        coord = WCSCoordinates(None)
        x, y = {}, {}
        with pytest.raises(TypeError):
            coord.world2pixel(x, y)

    def test_axis_label(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        assert coord.axis_label(0) == 'World 0: GLAT-TAN'
        assert coord.axis_label(1) == 'World 1: GLON-TAN'


class TestCoordinatesFromHeader(object):

    def test_2d(self):
        hdr = {"NAXIS" : 2}
        with patch('glue.core.coordinates.WCSCoordinates') as wcs:
            coord = coordinates_from_header(hdr)
            wcs.assert_called_once_with(hdr)

    def test_3d(self):
        hdr = {"NAXIS" : 3}
        with patch('glue.core.coordinates.WCSCubeCoordinates') as wcs:
            coord = coordinates_from_header(hdr)
            wcs.assert_called_once_with(hdr)

    def test_nod(self):
        hdr = {}
        with patch('glue.core.coordinates.Coordinates') as wcs:
            coord = coordinates_from_header(hdr)
            wcs.assert_called_once_with()

    def test_attribute_error(self):
        hdr = {"NAXIS" : 2}
        with patch('glue.core.coordinates.WCSCoordinates') as wcs:
            wcs.side_effect = AttributeError
            coord = coordinates_from_header(hdr)
            wcs.assert_called_once_with(hdr)
            assert type(coord) is Coordinates


HDR_2D_VALID = """SIMPLE  =                    T / Written by IDL:  Wed Jul 27 10:01:47 2011
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
