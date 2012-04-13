import unittest
import time

import numpy as np
import matplotlib.pyplot as plt

import glue

class TestRectangle(unittest.TestCase):

    def setUp(self):
        self.roi = glue.roi.RectangularROI()

    def test_empty(self):
        #self.assertFalse(self.roi.contains(1, 2))
        x = np.array([1,2])
        y = np.array([1,2])
        self.assertFalse(self.roi.contains(x, y).any())
        self.assertFalse(self.roi.contains(1, 2))

    def test_single(self):
        self.roi.update_limits(0, 0, 10, 10)
        self.assertTrue(self.roi.contains(5, 5))
        self.assertFalse(self.roi.contains(11, 11))

    def test_reset(self):
        self.assertFalse(self.roi.defined())
        self.roi.update_limits(0, 0, 10, 10)
        self.assertTrue(self.roi.defined())
        self.roi.reset()
        self.assertFalse(self.roi.contains(5, 5))
        self.assertFalse(self.roi.defined())

    def test_many(self):
        self.roi.update_limits(0, 0, 10, 10)

        x = np.array([5, 6, 2, 11])
        y = np.array([5, 11, 2, 11])
        result = self.roi.contains(x, y)
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])

class DummyEvent(object):
    def __init__(self, x, y, inaxes=True):
        self.inaxes = inaxes
        self.xdata = x
        self.ydata = y

class TestRectangleMpl(unittest.TestCase):

    def setUp(self):
        p = plt.plot([1,2,3])
        self.axes = p[0].axes
        self.roi = glue.roi.MplRectangularROI(self.axes)

    def assert_roi_correct(self, x0, x1, y0, y1):
        corner = self.roi.roi().corner()
        height = self.roi.roi().height()
        width = self.roi.roi().width()
        self.assertTrue(self.roi._roi.defined())
        self.assertAlmostEqual(corner[0], min(x0, x1), 4)
        self.assertAlmostEqual(corner[1], min(y0, y1), 4)
        self.assertAlmostEqual(abs(y1 - y0), height, 4)
        self.assertAlmostEqual(abs(x1 - x0), width, 4)
        self.assertTrue(self.roi._rectangle.get_visible())

        c2 = self.roi._rectangle.get_xy()
        w2 = self.roi._rectangle.get_width()
        h2 = self.roi._rectangle.get_height()
        self.assertAlmostEqual(corner[0], c2[0], 4)
        self.assertAlmostEqual(corner[1], c2[1], 4)
        self.assertAlmostEqual(width, w2, 4)
        self.assertAlmostEqual(height, h2, 4)

    def test_creation(self):
        self.assertFalse(self.roi._roi.defined())
        self.assertTrue(isinstance(self.roi._roi, glue.roi.RectangularROI))

    def test_define_roi_to_right(self):
        ev0 = DummyEvent(0.5, 0.5)
        ev1 = DummyEvent(1, 1)

        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)

        self.assert_roi_correct(.5, 1, .5, 1)

    def test_define_roi_to_left(self):
        ev0 = DummyEvent(1, 1)
        ev1 = DummyEvent(0.5, 0.5)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.assert_roi_correct(.5, 1, .5, 1)

if __name__ == "__main__":
    unittest.main()