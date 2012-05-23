import unittest

from pyfits import Header
import numpy as np

import glue


class TestWcsCoordinates(unittest.TestCase):

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
        coord = glue.coordinates.WCSCoordinates(hdr)

        x, y = 250., 187.5
        result = coord.pixel2world(x, y)
        expected = 0, 5
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])

    def test_pixel2world_different_input_types(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        x, y = 250, 187.5
        result = coord.pixel2world(x, y)
        expected = 0, 5
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])

    def test_pixel2world_list(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        x, y = [250, 250], [187.5, 187.5]
        result = coord.pixel2world(x, y)
        expected = [0, 0], [5, 5]
        for i in range(0,1):
            for r,e in zip(result[i], expected[i]):
                self.assertAlmostEqual(r, e)

    def test_pixel2world_numpy(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        x, y = np.array([250, 250]), np.array([187.5, 187.5])
        result = coord.pixel2world(x, y)
        expected = np.array([0, 0]), np.array([5, 5])

        np.testing.assert_array_almost_equal(result[0], expected[0])
        np.testing.assert_array_almost_equal(result[1], expected[1])

    def test_world2pixel_numpy(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        expected = np.array([250, 250]), np.array([187.5, 187.5])
        x, y = np.array([0, 0]), np.array([5, 5])
        result = coord.world2pixel(x, y)

        np.testing.assert_array_almost_equal(result[0], expected[0])
        np.testing.assert_array_almost_equal(result[1], expected[1])

    def test_world2pixel_list(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        expected = [250, 250], [187.5, 187.5]
        x, y = [0, 0], [5, 5]
        result = coord.world2pixel(x, y)
        for i in range(0,1):
            for r,e in zip(result[i], expected[i]):
                self.assertAlmostEqual(r, e)

    def test_world2pixel_scalar(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        expected = 250., 187.5
        x, y = 0, 5

        result = coord.world2pixel(x, y)
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])

    def test_world2pixel_different_input_types(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        expected = 250., 187.5
        x, y = 0, 5.

        result = coord.world2pixel(x, y)
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])

    def test_pixel2world_mismatched_input(self):
        coord = glue.coordinates.WCSCoordinates(None)
        x, y = 0, [5]
        self.assertRaises(TypeError, coord.pixel2world, x, y)

    def test_world2pixel_mismatched_input(self):
        coord = glue.coordinates.WCSCoordinates(None)
        x, y = 0, [5]
        self.assertRaises(TypeError, coord.world2pixel, x, y)

    def test_pixel2world_invalid_input(self):
        coord = glue.coordinates.WCSCoordinates(None)
        x, y = {}, {}
        self.assertRaises(TypeError, coord.pixel2world, x, y)

    def test_world2pixel_invalid_input(self):
        coord = glue.coordinates.WCSCoordinates(None)
        x, y = {}, {}
        self.assertRaises(TypeError, coord.world2pixel, x, y)

    def test_axis_label(self):
        hdr = self.default_header()
        coord = glue.coordinates.WCSCoordinates(hdr)

        self.assertEquals(coord.axis_label(0), 'World 0: GLAT-TAN')
        self.assertEquals(coord.axis_label(1), 'World 1: GLON-TAN')



if __name__ == "__main__":
    unittest.main()

