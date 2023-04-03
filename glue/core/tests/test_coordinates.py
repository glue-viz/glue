# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from glue.core.tests.test_state import clone
from glue.tests.helpers import requires_astropy

from ..coordinate_helpers import (axis_label, world_axis,
                                  pixel2world_single_axis, dependent_axes)
from ..coordinates import (coordinates_from_header, IdentityCoordinates,
                           WCSCoordinates, AffineCoordinates,
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
        result = coord.pixel_to_world_values(x, y)
        expected = 359.9832692105993601, 5.0166664867400375
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_different_input_types(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = 250, 187.5
        result = coord.pixel_to_world_values(x, y)
        expected = 359.9832692105993601, 5.0166664867400375
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_list(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = [250, 250], [187.5, 187.5]
        result = coord.pixel_to_world_values(x, y)
        expected = ([359.9832692105993601, 359.9832692105993601],
                    [5.0166664867400375, 5.0166664867400375])

        for i in range(0, 1):
            for r, e in zip(result[i], expected[i]):
                assert_allclose(r, e)

    def test_pixel2world_numpy(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = np.array([250, 250]), np.array([187.5, 187.5])
        result = coord.pixel_to_world_values(x, y)
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

        result = coord.world_to_pixel_values(x, y)
        np.testing.assert_array_almost_equal(result[0], expected[0], 3)
        np.testing.assert_array_almost_equal(result[1], expected[1], 3)

    def test_world2pixel_list(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        x, y = [0, 0], [0, 0]
        expected = ([249.0000000000000284, 249.0000000000000284],
                    [-114.2632689899972434, -114.2632689899972434])

        result = coord.world_to_pixel_values(x, y)
        for i in range(0, 1):
            for r, e in zip(result[i], expected[i]):
                assert_allclose(r, e)

    def test_world2pixel_scalar(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        expected = 249.0000000000000284, -114.2632689899972434
        x, y = 0, 0

        result = coord.world_to_pixel_values(x, y)
        assert_allclose(result[0], expected[0], 3)
        assert_allclose(result[1], expected[1], 3)

    def test_world2pixel_mismatched_input(self):
        coord = WCSCoordinates(self.default_header())
        x, y = 0., [0.]
        expected = coord.world_to_pixel_values(x, y[0])

        result = coord.world_to_pixel_values(x, y)
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_pixel2world_mismatched_input(self):
        coord = WCSCoordinates(self.default_header())
        x, y = [250.], 187.5
        expected = coord.pixel_to_world_values(x[0], y)

        result = coord.pixel_to_world_values(x, y)
        assert_allclose(result[0], expected[0])
        assert_allclose(result[1], expected[1])

    def test_axis_label(self):
        hdr = self.default_header()
        coord = WCSCoordinates(hdr)

        assert axis_label(coord, 0) == 'Galactic Latitude'
        assert axis_label(coord, 1) == 'Galactic Longitude'


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
    # pixel_axis and world_axis are in WCS order
    assert_allclose(world_axis(coord, data, pixel_axis=0, world_axis=0), [2, 0, -2, -4])
    assert_allclose(world_axis(coord, data, pixel_axis=1, world_axis=1), [5, 7, 9])


class TestCoordinatesFromHeader(object):

    def test_2d_nowcs(self):
        hdr = {"NAXIS": 2}
        coord = coordinates_from_header(hdr)
        assert type(coord) == IdentityCoordinates
        assert coord.pixel_n_dim == 2
        assert coord.world_n_dim == 2

    def test_2d(self):
        hdr = header_from_string(HDR_2D_VALID)
        coord = coordinates_from_header(hdr)
        assert type(coord) == WCSCoordinates

    def test_3d_nowcs(self):
        hdr = HDR_3D_VALID_NOWCS
        coord = coordinates_from_header(header_from_string(hdr))
        assert type(coord) == IdentityCoordinates
        assert coord.pixel_n_dim == 3
        assert coord.world_n_dim == 3

    def test_3d(self):
        hdr = header_from_string(HDR_3D_VALID_WCS)
        coord = coordinates_from_header(hdr)
        assert type(coord) == WCSCoordinates

    def test_nod(self):
        hdr = 0
        coord = coordinates_from_header(hdr)
        assert type(coord) == IdentityCoordinates


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
    result = coord.pixel_to_world_values(x, y)
    for r in result:
        assert r.shape == x.shape
    result = coord.world_to_pixel_values(x, y)
    for r in result:
        assert r.shape == x.shape

    x.shape = (4, 3)
    y.shape = (4, 3)
    result = coord.pixel_to_world_values(x, y)
    for r in result:
        assert r.shape == x.shape
    result = coord.world_to_pixel_values(x, y)
    for r in result:
        assert r.shape == x.shape

    x.shape = (2, 2, 3)
    y.shape = (2, 2, 3)
    result = coord.pixel_to_world_values(x, y)
    for r in result:
        assert r.shape == x.shape
    result = coord.world_to_pixel_values(x, y)
    for r in result:
        assert r.shape == x.shape


@requires_astropy
def test_coords_preserve_shape_3d():
    coord = coordinates_from_header(header_from_string(HDR_3D_VALID_NOWCS))
    x = np.zeros(12)
    y = np.zeros(12)
    z = np.zeros(12)
    result = coord.pixel_to_world_values(x, y, z)
    for r in result:
        assert r.shape == x.shape
    result = coord.world_to_pixel_values(x, y, z)
    for r in result:
        assert r.shape == x.shape

    x.shape = (4, 3)
    y.shape = (4, 3)
    z.shape = (4, 3)
    result = coord.pixel_to_world_values(x, y, z)
    for r in result:
        assert r.shape == x.shape
    result = coord.world_to_pixel_values(x, y, z)
    for r in result:
        assert r.shape == x.shape

    x.shape = (2, 2, 3)
    y.shape = (2, 2, 3)
    z.shape = (2, 2, 3)
    result = coord.pixel_to_world_values(x, y, z)
    for r in result:
        assert r.shape == x.shape
    result = coord.world_to_pixel_values(x, y, z)
    for r in result:
        assert r.shape == x.shape


def test_world_axis_units():
    coord = coordinates_from_header(header_from_string(HDR_3D_VALID_WCS))
    assert coord.world_axis_units[0] == 'deg'
    assert coord.world_axis_units[1] == 'deg'
    assert coord.world_axis_units[2] in ['m s-1', 'm.s**-1']


def test_dependent_axes_non_diagonal_pc():

    # Fix a bug that occurred when non-diagonal PC elements
    # were present in the WCS - in that case all other axes
    # were returned as dependent axes even if this wasn't
    # the case.

    coord = WCSCoordinates(naxis=3)
    coord.wcs.ctype = 'HPLN-TAN', 'HPLT-TAN', 'Time'
    coord.wcs.crval = 1, 1, 1
    coord.wcs.crpix = 1, 1, 1
    coord.wcs.cd = [[0.9, 0.1, 0], [-0.1, 0.9, 0], [0, 0, 1]]

    # Remember that the axes numbers below are reversed compared
    # to the WCS order above.
    assert_equal(dependent_axes(coord, 0), [0])
    assert_equal(dependent_axes(coord, 1), [1, 2])
    assert_equal(dependent_axes(coord, 2), [1, 2])


def test_pixel2world_single_axis():

    # Regression test for a bug in pixel2world_single_axis which was due to
    # incorrect indexing order (WCS vs Numpy)

    coord = WCSCoordinates(naxis=3)
    coord.wcs.ctype = 'HPLN-TAN', 'HPLT-TAN', 'Time'
    coord.wcs.crval = 1, 1, 1
    coord.wcs.crpix = 1, 1, 1
    coord.wcs.cd = [[0.9, 0.1, 0], [-0.1, 0.9, 0], [0, 0, 1]]

    x = np.array([0.2, 0.4, 0.6])
    y = np.array([0.3, 0.6, 0.9])
    z = np.array([0.5, 0.5, 0.5])

    assert_allclose(pixel2world_single_axis(coord, x, y, z, world_axis=0), [1.21004705, 1.42012044, 1.63021455])
    assert_allclose(pixel2world_single_axis(coord, x, y, z, world_axis=1), [1.24999002, 1.499947, 1.74985138])
    assert_allclose(pixel2world_single_axis(coord, x, y, z, world_axis=2), [1.5, 1.5, 1.5])


def test_pixel2world_single_axis_1d():

    # Regression test for issues that occurred for 1D WCSes

    coord = WCSCoordinates(naxis=1)
    coord.wcs.ctype = ['FREQ']
    coord.wcs.crpix = [1]
    coord.wcs.crval = [1]
    coord.wcs.cdelt = [1]

    x = np.array([0.2, 0.4, 0.6])
    expected = np.array([1.2, 1.4, 1.6])

    assert_allclose(pixel2world_single_axis(coord, x, world_axis=0), expected)
    assert_allclose(pixel2world_single_axis(coord, x.reshape((1, 3)), world_axis=0), expected.reshape((1, 3)))
    assert_allclose(pixel2world_single_axis(coord, x.reshape((3, 1)), world_axis=0), expected.reshape((3, 1)))


def test_pixel2world_single_axis_affine_1d():

    # Regression test for issues that occurred for 1D AffineCoordinates

    coord = AffineCoordinates(np.array([[2, 0], [0, 1]]))

    x = np.array([0.2, 0.4, 0.6])
    expected = np.array([0.4, 0.8, 1.2])

    assert_allclose(pixel2world_single_axis(coord, x, world_axis=0), expected)
    assert_allclose(pixel2world_single_axis(coord, x.reshape((1, 3)), world_axis=0), expected.reshape((1, 3)))
    assert_allclose(pixel2world_single_axis(coord, x.reshape((3, 1)), world_axis=0), expected.reshape((3, 1)))


def test_affine():

    matrix = np.array([[2, 3, -1], [1, 2, 2], [0, 0, 1]])

    coords = AffineCoordinates(matrix)

    assert axis_label(coords, 1) == 'World 1'
    assert axis_label(coords, 0) == 'World 0'

    assert coords.world_axis_units[1] == ''
    assert coords.world_axis_units[0] == ''

    # First the scalar case

    xp, yp = 1, 2

    xw, yw = coords.pixel_to_world_values(xp, yp)

    assert_allclose(xw, 2 * 1 + 3 * 2 - 1)
    assert_allclose(yw, 1 * 1 + 2 * 2 + 2)

    assert np.ndim(xw) == 0
    assert np.ndim(yw) == 0

    xpc, ypc = coords.world_to_pixel_values(xw, yw)

    assert_allclose(xpc, 1)
    assert_allclose(ypc, 2)

    assert np.ndim(xpc) == 0
    assert np.ndim(ypc) == 0

    # Next the array case

    xp = np.array([1, 2, 3])
    yp = np.array([2, 3, 4])

    xw, yw = coords.pixel_to_world_values(xp, yp)

    assert_allclose(xw, 2 * xp + 3 * yp - 1)
    assert_allclose(yw, 1 * xp + 2 * yp + 2)

    xpc, ypc = coords.world_to_pixel_values(xw, yw)

    assert_allclose(xpc, [1, 2, 3])
    assert_allclose(ypc, [2, 3, 4])

    # Check that serialization/deserialization works

    coords2 = clone(coords)

    xw, yw = coords2.pixel_to_world_values(xp, yp)

    assert_allclose(xw, 2 * xp + 3 * yp - 1)
    assert_allclose(yw, 1 * xp + 2 * yp + 2)

    xpc, ypc = coords.world_to_pixel_values(xw, yw)

    assert_allclose(xpc, [1, 2, 3])
    assert_allclose(ypc, [2, 3, 4])


def test_affine_labels_units():

    matrix = np.array([[2, 3, -1], [1, 2, 2], [0, 0, 1]])

    coords = AffineCoordinates(matrix, units=['km', 'km'], labels=['xw', 'yw'])

    assert axis_label(coords, 1) == 'Xw'
    assert axis_label(coords, 0) == 'Yw'

    assert coords.world_axis_units[1] == 'km'
    assert coords.world_axis_units[0] == 'km'

    coords2 = clone(coords)

    assert axis_label(coords2, 1) == 'Xw'
    assert axis_label(coords2, 0) == 'Yw'

    assert coords2.world_axis_units[1] == 'km'
    assert coords2.world_axis_units[0] == 'km'


def test_affine_invalid():

    matrix = np.array([[2, 3, -1], [1, 2, 2], [0, 0, 1]])

    with pytest.raises(ValueError) as exc:
        AffineCoordinates(matrix[0])
    assert exc.value.args[0] == 'Affine matrix should be two-dimensional'

    with pytest.raises(ValueError) as exc:
        AffineCoordinates(matrix[:-1])
    assert exc.value.args[0] == 'Affine matrix should be square'

    with pytest.raises(ValueError) as exc:
        AffineCoordinates(matrix, labels=['a', 'b', 'c'])
    assert exc.value.args[0] == 'Expected 2 labels, got 3'

    with pytest.raises(ValueError) as exc:
        AffineCoordinates(matrix, units=['km', 'km', 'km'])
    assert exc.value.args[0] == 'Expected 2 units, got 3'

    matrix[-1] = 1

    with pytest.raises(ValueError) as exc:
        AffineCoordinates(matrix)
    assert exc.value.args[0] == 'Last row of matrix should be zeros and a one'


def test_affine_highd():

    # Test AffineCoordinates when higher dimensional objects are transformed

    matrix = np.array([[2, 3, -1], [1, 2, 2], [0, 0, 1]])
    coords = AffineCoordinates(matrix)

    xp = np.ones((2, 4, 1, 2, 5))
    yp = np.ones((2, 4, 1, 2, 5))

    xw, yw = coords.pixel_to_world_values(xp, yp)
    xpc, ypc = coords.world_to_pixel_values(xw, yw)

    assert_allclose(xp, xpc)
    assert_allclose(yp, ypc)
