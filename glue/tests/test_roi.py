import unittest
import time

import numpy as np
import matplotlib.pyplot as plt

import glue

class TestRectangle(unittest.TestCase):

    def setUp(self):
        self.roi = glue.roi.RectangularROI()

    def test_empty(self):
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 1, 2)
        x = np.array([1,2])
        y = np.array([1,2])
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, x, y)

    def test_single(self):
        self.roi.update_limits(0, 0, 10, 10)
        self.assertTrue(self.roi.contains(5, 5))
        self.assertFalse(self.roi.contains(11, 11))

    def test_reset(self):
        self.assertFalse(self.roi.defined())
        self.roi.update_limits(0, 0, 10, 10)
        self.assertTrue(self.roi.defined())
        self.roi.reset()
        self.assertFalse(self.roi.defined())
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 5, 5)

    def test_poly(self):
        self.roi.update_limits(0, 0, 10, 10)
        x,y = self.roi.to_polygon()
        poly = glue.roi.PolygonalROI(vx = x, vy = y)
        self.assertTrue(poly.contains(5, 5))

    def test_many(self):
        self.roi.update_limits(0, 0, 10, 10)

        x = np.array([5, 6, 2, 11])
        y = np.array([5, 11, 2, 11])
        result = self.roi.contains(x, y)
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])

    def test_multidim(self):
        self.roi.update_limits(0,0,10,10)
        x = np.array([1, 2, 3, 4]).reshape(2,2)
        y = np.array([1, 2, 3, 4]).reshape(2,2)
        self.assertTrue(self.roi.contains(x,y).all())
        self.assertFalse(self.roi.contains(x+10,y).any())
        self.assertEquals(self.roi.contains(x,y).shape, x.shape)

class TestCircle(unittest.TestCase):
    def setUp(self):
        self.roi = glue.roi.CircularROI()

    def test_empty(self):
        self.assertFalse(self.roi.defined())
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 1, 1)

    def test_set_center(self):
        self.roi.set_center(0, 0)
        self.roi.set_radius(1)
        self.assertTrue(self.roi.contains(0,0))
        self.assertFalse(self.roi.contains(2,2))
        self.roi.set_center(2,2)
        self.assertFalse(self.roi.contains(0,0))
        self.assertTrue(self.roi.contains(2,2))

    def test_set_radius(self):
        self.roi.set_center(0,0)
        self.roi.set_radius(1)
        self.assertFalse(self.roi.contains(1.5,0))
        self.roi.set_radius(5)
        self.assertTrue(self.roi.contains(1.5, 0))

    def test_contains_many(self):
        x = [0,0,0,0,0]
        y = [0,0,0,0,0]
        self.roi.set_center(0,0)
        self.roi.set_radius(1)
        self.assertTrue(all(self.roi.contains(x, y)))
        self.assertTrue(all(self.roi.contains(np.asarray(x), np.asarray(y))))
        self.assertFalse(any(self.roi.contains(np.asarray(x)+10, y)))

    def test_poly(self):
        self.roi.set_center(0,0)
        self.roi.set_radius(1)
        x,y = self.roi.to_polygon()
        poly = glue.roi.PolygonalROI(vx=x, vy=y)
        self.assertTrue(poly.contains(0,0))
        self.assertFalse(poly.contains(10,0))

    def test_reset(self):
        self.assertFalse(self.roi.defined())
        self.roi.set_center(0,0)
        self.assertFalse(self.roi.defined())
        self.roi.set_radius(2)
        self.assertTrue(self.roi.defined())
        self.roi.reset()
        self.assertFalse(self.roi.defined())
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 0, 0)

    def test_multidim(self):
        self.roi.set_center(0,0)
        self.roi.set_radius(1)
        x = np.array([.1, .2, .3, .4]).reshape(2,2)
        y = np.array([-.1, -.2, -.3, -.4]).reshape(2,2)
        self.assertTrue(self.roi.contains(x,y).all())
        self.assertFalse(self.roi.contains(x+1, y).any())
        self.assertEquals(self.roi.contains(x,y).shape, (2,2))


class TestPolygon(unittest.TestCase):
    def setUp(self):
        self.roi = glue.roi.PolygonalROI()

    def define_as_square(self):
        self.roi.reset()
        self.assertFalse(self.roi.defined())
        self.roi.add_point(0,0)
        self.roi.add_point(0, 1)
        self.roi.add_point(1, 1)
        self.roi.add_point(1, 0)
        self.assertTrue(self.roi.defined())

    def test_replace(self):
        self.define_as_square()
        self.assertTrue(self.roi.contains(.9, .02))
        self.roi.replace_last_point(0,0)
        self.assertFalse(self.roi.contains(.9, .01))

    def test_remove(self):

        #successful remove
        self.define_as_square()
        self.assertTrue(self.roi.contains(.9, .02))
        self.roi.remove_point(1, 0)
        self.assertFalse(self.roi.contains(.9, .01))

        #failed remove
        self.define_as_square()
        self.assertTrue(self.roi.contains(.9, .02))
        self.roi.remove_point(1.5, 0, thresh = .49)
        self.assertTrue(self.roi.contains(.9, .01))

    def test_to_poly(self):
        self.define_as_square()
        x,y = self.roi.to_polygon()
        self.assertEquals(x[0], 0)
        self.assertEquals(x[1], 0)
        self.assertEquals(x[2], 1)
        self.assertEquals(x[3], 1)
        self.assertEquals(y[0], 0)
        self.assertEquals(y[1], 1)
        self.assertEquals(y[2], 1)
        self.assertEquals(y[3], 0)

    def test_rect(self):
        self.roi.reset()
        self.roi.add_point(4.95164474584, 0.136922625654)
        self.roi.add_point(6.08256580223, 0.136922625654)
        self.roi.add_point(6.08256580223, 0.423771697842)
        self.roi.add_point(4.95164474584, 0.423771697842)
        self.roi.add_point(4.95164474584, 0.136922625654)
        x = np.array([4.4, 3.18, 4.49, 4.49])
        y = np.array([.2, .2, .2, .2])
        self.assertFalse(self.roi.contains(4.4000001, 0.2))
        self.assertFalse(self.roi.contains(3.1800001, 0.2))
        self.assertFalse(self.roi.contains(4.4899998, 0.2))
        self.assertFalse(self.roi.contains(x, y).any())
        x.shape = (2,2)
        y.shape = (2,2)
        self.assertFalse(self.roi.contains(x,y).any())
        self.assertEquals(self.roi.contains(x,y).shape, (2,2))

    def test_empty(self):
        self.assertFalse(self.roi.defined())
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 0, 0)

    def test_contains_scalar(self):
        self.define_as_square()
        self.assertTrue(self.roi.contains(.5, .5))
        self.assertFalse(self.roi.contains(1.5, 1.5))

    def test_contains_list(self):
        self.define_as_square()
        self.assertTrue(self.roi.contains([.5, .4], [.5, .4]).all())
        self.assertFalse(self.roi.contains([1.5, 1.5], [0,0]).any())

    def test_contains_numpy(self):
        self.define_as_square()
        x = np.array([.4, .5, .4])
        y = np.array([.4, .5, .4])
        self.assertTrue(self.roi.contains(x, y).all())
        self.assertFalse(self.roi.contains(x+5, y).any())


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

    def tearDown(self):
        plt.close('all')

    def assert_roi_correct(self, x0, x1, y0, y1):
        corner = self.roi.roi().corner()
        height = self.roi.roi().height()
        width = self.roi.roi().width()
        self.assertTrue(self.roi._roi.defined())
        self.assertAlmostEqual(corner[0], min(x0, x1), 4)
        self.assertAlmostEqual(corner[1], min(y0, y1), 4)
        self.assertAlmostEqual(abs(y1 - y0), height, 4)
        self.assertAlmostEqual(abs(x1 - x0), width, 4)
        self.assertFalse(self.roi._rectangle.get_visible())

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
        self.roi.finalize_selection(ev1)
        self.assert_roi_correct(.5, 1, .5, 1)

    def test_define_roi_to_left(self):
        ev0 = DummyEvent(1, 1)
        ev1 = DummyEvent(0.5, 0.5)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)
        self.assert_roi_correct(.5, 1, .5, 1)

class TestCircleMpl(unittest.TestCase):
    def setUp(self):
        p = plt.plot([1,2,3])
        self.axes = p[0].axes
        self.roi = glue.roi.MplCircularROI(self.axes)

    def tearDown(self):
        plt.close('all')

    def test_creation(self):
        self.assertFalse(self.roi._roi.defined())
        self.assertTrue(isinstance(self.roi._roi, glue.roi.CircularROI))

    def send_events(self):
        """ Simulates mouse clicks at (x,y) = 100,100 pixels,
        drag to 100,150, and release."""
        xp1, yp1 = [100], [100]
        xp2, yp2 = [100], [150]
        xyd1 = glue.roi.pixel_to_data(self.axes, xp1, yp1)
        xyd2 = glue.roi.pixel_to_data(self.axes, xp2, yp2)
        ev0 = DummyEvent(xyd1[0,0], xyd1[0,1])
        ev1 = DummyEvent(xyd2[0,0], xyd2[0,1])
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)

    def assert_roi_correct(self):
        """ Tests that the ROI correctly describes a circle in
        pixel coordinates, defined via the events in send_events()"""
        xp1, yp1 = [100], [149]
        xp2, yp2 = [100], [151]
        xyd1 = glue.roi.pixel_to_data(self.axes, xp1, yp1)
        xyd2 = glue.roi.pixel_to_data(self.axes, xp2, yp2)
        roi = self.roi.roi()
        self.assertTrue(roi.contains(xyd1[0,0], xyd1[0,1]))
        self.assertFalse(roi.contains(xyd2[0,0], xyd2[0,1]))

    def test_define_xlog(self):
        self.axes.set_xscale('log')
        self.axes.set_xlim([1, 10])
        self.axes.set_ylim([0, 3])
        self.send_events()
        self.assert_roi_correct()

    def test_define_ylog(self):
        self.axes.set_yscale('log')
        self.axes.set_xlim([1, 10])
        self.axes.set_ylim([1, 3])
        self.send_events()
        self.assert_roi_correct()

    def test_define_loglog(self):
        self.axes.set_xscale('log')
        self.axes.set_yscale('log')
        self.axes.set_xlim([1, 10])
        self.axes.set_ylim([1, 3])
        self.send_events()
        self.assert_roi_correct()

    def test_define_loglog(self):
        self.axes.set_xscale('linear')
        self.axes.set_yscale('linear')
        self.axes.set_xlim([1, 10])
        self.axes.set_ylim([1, 3])
        self.send_events()
        self.assert_roi_correct()

class TestPolyMpl(unittest.TestCase):
    def setUp(self):
        p = plt.plot([1,2,3])
        self.axes = p[0].axes
        self.roi = glue.roi.MplPolygonalROI(self.axes)

    def tearDown(self):
        plt.close('all')

    def send_events(self):
        ev0 = DummyEvent(0, 0)
        ev1 = DummyEvent(0, 1)
        ev2 = DummyEvent(1, 1)
        ev3 = DummyEvent(1, 0)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.update_selection(ev2)
        self.roi.update_selection(ev3)
        self.roi.finalize_selection(ev3)

    def assert_roi_correct(self):
        roi = self.roi.roi()
        self.assertTrue(roi.contains(.5, .5))
        self.assertFalse(roi.contains(1.5, .5))

    def test_create(self):
        self.assertFalse(self.roi._roi.defined())
        self.assertTrue(isinstance(self.roi._roi, glue.roi.PolygonalROI))

    def test_define(self):
        self.send_events()
        self.assert_roi_correct()


class TestUtil(unittest.TestCase):
    def setUp(self):
        p = plt.plot([1,2,3])
        self.axes = p[0].axes

    def tearDown(self):
        plt.close('all')

    def test_aspect_ratio(self):
        self.axes.figure.set_figheight(5)
        self.axes.figure.set_figwidth(5)
        self.axes.set_xlim([0, 10])
        self.axes.set_ylim([0, 10])

        ax0 = glue.roi.aspect_ratio(self.axes)
        self.axes.set_ylim(0, 20)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0 / 2, 4)
        self.axes.set_ylim(0, 5)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0 * 2, 4)
        self.axes.set_xlim(0, 5)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0, 4)
        self.axes.set_xlim(0, 10)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0 * 2, 4)

    def test_data_to_norm(self):
        self.axes.set_xlim(0,10)
        self.axes.set_ylim(0,10)
        func = glue.roi.data_to_norm
        self.assertAlmostEquals(func(self.axes, 0, 0)[0,0], 0, 3)
        self.assertAlmostEquals(func(self.axes, 0, 0)[0,1], 0, 3)
        self.assertAlmostEquals(func(self.axes, 0, 10)[0,0], 0, 3)
        self.assertAlmostEquals(func(self.axes, 0, 10)[0,1], 1, 3)
        self.assertAlmostEquals(func(self.axes, 10, 10)[0,0], 1, 3)
        self.assertAlmostEquals(func(self.axes, 10, 10)[0,1], 1, 3)
        self.assertAlmostEquals(func(self.axes, 10, 0)[0,0], 1, 3)
        self.assertAlmostEquals(func(self.axes, 10, 0)[0,1], 0, 3)

        x = np.array([0, 0, 10, 10])
        y = np.array([0, 10, 0, 10])
        xans = [0, 0, 1, 1]
        yans = [0, 1, 0, 1]
        norm = func(self.axes, x, y)
        self.assertAlmostEquals(norm[0,0], xans[0], 3)
        self.assertAlmostEquals(norm[1,0], xans[1], 3)
        self.assertAlmostEquals(norm[2,0], xans[2], 3)
        self.assertAlmostEquals(norm[3,0], xans[3], 3)
        self.assertAlmostEquals(norm[0,1], yans[0], 3)
        self.assertAlmostEquals(norm[1,1], yans[1], 3)
        self.assertAlmostEquals(norm[2,1], yans[2], 3)
        self.assertAlmostEquals(norm[3,1], yans[3], 3)

        x = [0, 0, 10, 10]
        y = [0, 10, 0, 10]
        self.assertAlmostEquals(norm[0,0], xans[0], 3)
        self.assertAlmostEquals(norm[1,0], xans[1], 3)
        self.assertAlmostEquals(norm[2,0], xans[2], 3)
        self.assertAlmostEquals(norm[3,0], xans[3], 3)
        self.assertAlmostEquals(norm[0,1], yans[0], 3)
        self.assertAlmostEquals(norm[1,1], yans[1], 3)
        self.assertAlmostEquals(norm[2,1], yans[2], 3)
        self.assertAlmostEquals(norm[3,1], yans[3], 3)


    def test_data_to_pixel(self):
        xp = 100
        yp = 100
        data = glue.roi.pixel_to_data(self.axes, xp, yp)
        pixel = glue.roi.data_to_pixel(self.axes, data[:,0], data[:,1])
        self.assertAlmostEquals(pixel[0,0], xp, 3)
        self.assertAlmostEquals(pixel[0,1], yp, 3)

if __name__ == "__main__":
    unittest.main()