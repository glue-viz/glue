import unittest
import time

import numpy as np
import matplotlib.pyplot as plt
from mock import MagicMock

import glue

AXES = plt.plot([1,2,3])[0].axes

class TestRectangle(unittest.TestCase):

    def setUp(self):
        self.roi = glue.roi.RectangularROI()

    def test_empty_roi_contains_raises(self):
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 1, 2)

    def test_scalar_contains(self):
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

    def test_empty_to_polygon(self):
        x, y = self.roi.to_polygon()
        self.assertEquals(x, [])
        self.assertEquals(y, [])

    def test_to_polygon(self):
        self.roi.update_limits(0, 0, 10, 10)
        x,y = self.roi.to_polygon()
        poly = glue.roi.PolygonalROI(vx = x, vy = y)
        self.assertTrue(poly.contains(5, 5))

    def test_ndarray(self):
        self.roi.update_limits(0, 0, 10, 10)

        x = np.array([5, 6, 2, 11])
        y = np.array([5, 11, 2, 11])
        result = self.roi.contains(x, y)
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])
        self.assertFalse(result[3])

    def test_corner(self):
        self.roi.update_limits(6, 7, 10, 10)
        self.assertEquals(self.roi.corner(), (6,7))

    def test_width(self):
        self.roi.update_limits(2, 2, 10, 12)
        self.assertEquals(self.roi.width(), 8)

    def test_height(self):
        self.roi.update_limits(2, 2, 10, 12)
        self.assertEquals(self.roi.height(), 10)

    def test_multidim_ndarray(self):
        self.roi.update_limits(0,0,10,10)
        x = np.array([1, 2, 3, 4]).reshape(2,2)
        y = np.array([1, 2, 3, 4]).reshape(2,2)
        self.assertTrue(self.roi.contains(x,y).all())
        self.assertFalse(self.roi.contains(x+10,y).any())
        self.assertEquals(self.roi.contains(x,y).shape, x.shape)

    def test_str_undefined(self):
        """ str method should not crash """
        self.assertEquals(type(str(self.roi)), str)

    def test_str_defined(self):
        """ str method should not crash """
        self.roi.update_limits(1, 2, 3, 4)
        self.assertEquals(type(str(self.roi)), str)


class TestCircle(unittest.TestCase):
    def setUp(self):
        self.roi = glue.roi.CircularROI()

    def test_undefined_on_creation(self):
        self.assertFalse(self.roi.defined())

    def test_contains_on_undefined_contains_raises(self):
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

    def test_poly_undefined(self):
        x, y = self.roi.to_polygon()
        self.assertEquals(x, [])
        self.assertEquals(y, [])

    def test_reset(self):
        self.assertFalse(self.roi.defined())
        self.roi.set_center(0,0)
        self.assertFalse(self.roi.defined())
        self.roi.set_radius(2)
        self.assertTrue(self.roi.defined())
        self.roi.reset()
        self.assertFalse(self.roi.defined())

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

    def test_contains_on_empty_raises(self):
        self.assertRaises(glue.exceptions.UndefinedROI,
                          self.roi.contains, 1, 2)

    def test_remove_empty(self):
        self.roi.remove_point(1, 0)

    def test_replace(self):
        self.define_as_square()
        self.assertTrue(self.roi.contains(.9, .02))
        self.roi.replace_last_point(0,0)
        self.assertFalse(self.roi.contains(.9, .01))

    def test_remove_successful(self):
        self.define_as_square()
        self.assertTrue(self.roi.contains(.9, .02))
        self.roi.remove_point(1, 0)
        self.assertFalse(self.roi.contains(.9, .01))

    def test_remove_unsuccessful(self):
        self.define_as_square()
        self.assertTrue(self.roi.contains(.9, .02))
        self.roi.remove_point(1.5, 0, thresh = .49)
        self.assertTrue(self.roi.contains(.9, .01))

    def test_to_poly(self):
        self.define_as_square()
        x,y = self.roi.to_polygon()
        self.assertEquals(x, [0,0,1,1])
        self.assertEquals(y, [0,1,1,0])

    def test_to_poly_undefined(self):
        self.assertEquals(self.roi.to_polygon(), ([], []))

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

    def test_str(self):
        """ __str__ returns a string """
        self.assertEquals(type(str(self.roi)), str)


class DummyEvent(object):
    def __init__(self, x, y, inaxes=True):
        self.inaxes = inaxes
        self.xdata = x
        self.ydata = y

class MockAxes(object):
    def __init__(self):
        self.figure = MagicMock()
        self.figure.canvas = MagicMock()

    def add_patch(self, patch):
        pass


class TestMpl(unittest.TestCase):

    def setUp(self):
        self.axes = MagicMock()
        self.roi = self._roi_factory()

    def _roi_factory(self):
        raise NotImplemented

    def test_undefined_on_creation(self):
        self.assertFalse(self.roi._roi.defined())

    def test_proper_roi(self):
        raise NotImplemented

    def test_start_ignored_if_not_inaxes(self):
        ev = DummyEvent(0, 0, inaxes=False)
        self.roi.start_selection(ev)
        self.assertFalse(self.roi._roi.defined())

    def test_canvas_syncs_properly(self):
        self.assertEquals(self.axes.figure.canvas.draw.call_count, 1)
        event = DummyEvent(5,5)
        self.roi.start_selection(event)
        self.assertEquals(self.axes.figure.canvas.draw.call_count, 2)
        self.roi.update_selection(event)
        self.assertEquals(self.axes.figure.canvas.draw.call_count, 3)
        self.roi.update_selection(event)
        self.assertEquals(self.axes.figure.canvas.draw.call_count, 4)
        self.roi.finalize_selection(event)
        self.assertEquals(self.axes.figure.canvas.draw.call_count, 5)

    def test_patch_shown_on_start(self):
        self.assertFalse(self.roi._patch.get_visible())
        event = DummyEvent(5, 5)
        self.roi.start_selection(event)
        self.assertTrue(self.roi._patch.get_visible())

    def test_patch_hidden_on_finalise(self):
        event = DummyEvent(5, 5)
        self.roi.start_selection(event)
        self.roi.finalize_selection(event)
        self.assertFalse(self.roi._patch.get_visible())

    def test_update_before_start_ignored(self):
        self.roi.update_selection(None)
        self.assertFalse(self.roi._roi.defined())

    def test_finalize_before_start_ignored(self):
        self.roi.finalize_selection(None)
        self.assertFalse(self.roi._roi.defined())

    def test_roi_defined_after_start(self):
        event = DummyEvent(5, 5)
        self.roi.start_selection(event)
        self.assertTrue(self.roi._roi.defined())

    def test_roi_undefined_before_start(self):
        self.assertFalse(self.roi._roi.defined())


class TestRectangleMpl(TestMpl):

    def _roi_factory(self):
        return glue.roi.MplRectangularROI(self.axes)

    def assert_roi_correct(self, x0, x1, y0, y1):
        corner = self.roi.roi().corner()
        height = self.roi.roi().height()
        width = self.roi.roi().width()
        self.assertTrue(self.roi.roi().defined())
        self.assertAlmostEqual(corner[0], min(x0, x1), 4)
        self.assertAlmostEqual(corner[1], min(y0, y1), 4)
        self.assertAlmostEqual(abs(y1 - y0), height, 4)
        self.assertAlmostEqual(abs(x1 - x0), width, 4)

    def assert_patch_correct(self, x0, x1, y0, y1):
        corner = self.roi._patch.get_xy()
        width = self.roi._patch.get_width()
        height = self.roi._patch.get_height()
        self.assertAlmostEqual(corner, (min(x0, x1), min(y0, y1)), 4)
        self.assertAlmostEqual(width, abs(x0-x1))
        self.assertAlmostEqual(height, abs(y0-y1))

    def test_str(self):
        self.assertEquals(type(str(self.roi)), str)

    def test_proper_roi(self):
        self.assertTrue(isinstance(self.roi._roi, glue.roi.RectangularROI))

    def test_roi_on_start_selection(self):
        event = DummyEvent(5, 5)
        self.roi.start_selection(event)
        self.assert_roi_correct(5,5,5,5)

    def test_patch_on_start_selection(self):
        event = DummyEvent(5, 5)
        self.roi.start_selection(event)
        self.assert_patch_correct(5,5,5,5)


    def test_update_selection(self):
        event = DummyEvent(5, 6)
        self.roi.start_selection(event)
        event = DummyEvent(10, 11)
        self.roi.update_selection(event)
        self.assert_roi_correct(5,10,6,11)
        self.assert_patch_correct(5,10,6,11)

    def test_finalize_selection(self):
        event = DummyEvent(5, 6)
        self.roi.start_selection(event)
        event = DummyEvent(10, 11)
        self.roi.update_selection(event)
        self.roi.finalize_selection(event)
        self.assert_roi_correct(5,10,6,11)
        self.assert_patch_correct(5,10,6,11)

    def test_define_roi_to_right(self):
        ev0 = DummyEvent(0.5, 0.5)
        ev1 = DummyEvent(1, 1)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)

        self.assert_roi_correct(.5, 1, .5, 1)
        self.assert_patch_correct(.5, 1, .5, 1)

    def test_define_roi_to_left(self):
        ev0 = DummyEvent(1, 1)
        ev1 = DummyEvent(0.5, 0.5)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)

        self.assert_roi_correct(.5, 1, .5, 1)
        self.assert_patch_correct(.5, 1, .5, 1)

class TestCircleMpl(TestMpl):
    def _roi_factory(self):
        return glue.roi.MplCircularROI(self.axes)

    def setUp(self):
        super(TestCircleMpl, self).setUp()
        self.pixel_to_data = glue.roi.pixel_to_data
        self.data_to_pixel = glue.roi.data_to_pixel

        glue.roi.pixel_to_data = lambda x,y,z: np.column_stack((y,z))
        glue.roi.data_to_pixel = lambda x,y,z: np.column_stack((y,z))

    def tearDown(self):
        # restore methods
        glue.roi.pixel_to_data = self.pixel_to_data
        glue.roi.data_to_pixel = self.data_to_pixel

    def test_proper_roi(self):
        self.assertTrue(isinstance(self.roi._roi, glue.roi.CircularROI))

    def test_to_polygon_undefined(self):
        """to_polygon() result should be undefined before defining polygon"""
        roi = self.roi.roi()
        self.assertFalse(roi.defined())


    def test_roi_defined_correctly(self):
        ev0 = DummyEvent(0, 0)
        ev1 = DummyEvent(5, 0)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)
        self.assert_roi_correct(0, 0, 5)

    def assert_roi_correct(self, x, y, r):
        roi = self.roi.roi()
        self.assertTrue(roi.defined())
        self.assertTrue(roi.contains(x, y))
        self.assertTrue(roi.contains(x + .95 * r, y))
        self.assertFalse(roi.contains(x + 1.05 * r, y))
        self.assertFalse(roi.contains(x + .8 * r, y + .8 * r))


class TestPolyMpl(TestMpl):
    def _roi_factory(self):
        return glue.roi.MplPolygonalROI(self.axes)

    def test_proper_roi(self):
        return isinstance(self.roi._roi, glue.roi.PolygonalROI)

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

    def test_define(self):
        self.send_events()
        self.assert_roi_correct()


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.axes = AXES

    def test_aspect_ratio(self):
        self.axes.figure.set_figheight(5)
        self.axes.figure.set_figwidth(5)
        self.axes.set_ylim([0,10])
        self.axes.set_xlim([0,10])

        ax0 = glue.roi.aspect_ratio(self.axes)
        self.axes.set_ylim(0, 20)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0 / 2, 4)
        self.axes.set_ylim(0, 5)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0 * 2, 4)
        self.axes.set_xlim(0, 5)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0, 4)
        self.axes.set_xlim(0, 10)
        self.assertAlmostEquals(glue.roi.aspect_ratio(self.axes), ax0 * 2, 4)

    def test_data_to_norm_with_scalars(self):
        # assume data that gets submitted to axes is valid.
        # testing to see if we can get there
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

del TestMpl # prevents unittest discovery from running abstract base class

if __name__ == "__main__":
    unittest.main()