# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from collections import namedtuple

import pytest
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from matplotlib.figure import Figure
from numpy.testing import assert_almost_equal

from .. import roi as r
from ..component import CategoricalComponent
from ..roi import (RectangularROI, UndefinedROI, CircularROI, PolygonalROI, CategoricalROI,
                   MplCircularROI, MplRectangularROI, MplPolygonalROI, MplPickROI, PointROI,
                   XRangeROI, MplXRangeROI, YRangeROI, MplYRangeROI, RangeROI, Projected3dROI,
                   EllipticalROI)
from ..state import GlueSerializer, GlueUnSerializer


FIG = Figure()
AXES = FIG.add_subplot(111)


def roundtrip_roi(roi):
    gs = GlueSerializer(roi)
    out_str = gs.dumps()
    obj = GlueUnSerializer.loads(out_str)
    return obj.object('__main__')


class TestPoint(object):

    def setup_method(self, method):
        self.roi = PointROI(1, 2)

    def test_contains(self):
        assert not self.roi.contains(3, 3)
        assert not self.roi.contains(1, 2)

    def test_move_to(self):
        self.roi.move_to(4, 5)
        assert self.roi.x == 4
        assert self.roi.y == 5

    def test_defined(self):
        assert self.roi.defined()

    def test_not_defined(self):
        self.roi.reset()
        assert not self.roi.defined()

    def test_center(self):
        assert self.roi.center() == (1, 2)


class TestRectangle(object):

    def setup_method(self, method):
        self.roi = RectangularROI()

    def test_empty_roi_contains_raises(self):
        with pytest.raises(UndefinedROI):
            self.roi.contains(1, 2)

    def test_scalar_contains(self):
        self.roi.update_limits(0, 0, 10, 10)
        assert self.roi.contains(5, 5)
        assert not self.roi.contains(11, 11)

    def test_reset(self):
        assert not self.roi.defined()
        self.roi.update_limits(0, 0, 10, 10)
        assert self.roi.defined()
        self.roi.reset()
        assert not self.roi.defined()
        with pytest.raises(UndefinedROI):
            self.roi.contains(5, 5)

    def test_set_rotation(self):
        self.roi.update_limits(0, 0, 10, 4)
        self.roi.rotate_by(np.pi / 6)
        assert self.roi.contains(8, 6)
        self.roi.rotate_by(np.pi / 3)
        assert not self.roi.contains(9, 6)
        assert self.roi.contains(6, 6.9)
        self.roi.rotate_to(-np.pi / 3)
        assert not self.roi.contains(5, 6)
        assert self.roi.contains(6, -3)

    def test_empty_to_polygon(self):
        x, y = self.roi.to_polygon()
        assert x == []
        assert y == []

    def test_to_polygon(self):
        self.roi.update_limits(0, 0, 10, 10)
        x, y = self.roi.to_polygon()
        poly = PolygonalROI(vx=x, vy=y)
        assert poly.contains(5, 5)
        self.roi.update_limits(0, 0, 10, 4)
        self.roi.rotate_to(np.pi / 3)
        x, y = self.roi.to_polygon()
        poly = PolygonalROI(vx=x, vy=y)
        assert not poly.contains(8, 3)
        assert poly.contains(6, 6)

    def test_ndarray(self):
        self.roi.update_limits(0, 0, 10, 10)

        x = np.array([5, 6, 2, 11])
        y = np.array([5, 11, 2, 11])
        result = self.roi.contains(x, y)
        assert result[0]
        assert not result[1]
        assert result[2]
        assert not result[3]

    def test_corner(self):
        self.roi.update_limits(6, 7, 10, 10)
        assert self.roi.corner() == (6, 7)

    def test_width(self):
        self.roi.update_limits(2, 2, 10, 12)
        assert self.roi.width() == 8

    def test_height(self):
        self.roi.update_limits(2, 2, 10, 12)
        assert self.roi.height() == 10

    def test_multidim_ndarray(self):
        self.roi.update_limits(0, 0, 10, 10)
        x = np.array([1, 2, 3, 4]).reshape(2, 2)
        y = np.array([1, 2, 3, 4]).reshape(2, 2)
        assert self.roi.contains(x, y).all()
        assert not self.roi.contains(x + 10, y).any()
        assert self.roi.contains(x, y).shape == x.shape

    def test_str_undefined(self):
        """ str method should not crash """
        assert type(str(self.roi)) == str

    def test_str_defined(self):
        """ str method should not crash """
        self.roi.update_limits(1, 2, 3, 4)
        assert type(str(self.roi)) == str

    def test_serialization(self):
        self.roi.update_limits(1, 2, 3, 4)
        new_roi = roundtrip_roi(self.roi)
        assert_almost_equal(new_roi.xmin, 1)
        assert_almost_equal(new_roi.ymin, 2)
        assert_almost_equal(new_roi.xmax, 3)
        assert_almost_equal(new_roi.ymax, 4)


class TestRange(object):

    def test_wrong_orientation(self):
        with pytest.raises(ValueError):
            RangeROI(orientation='a')


class TestXRange(object):

    def test_undefined_on_init(self):
        assert not XRangeROI().defined()

    def test_str(self):
        roi = XRangeROI()
        assert str(roi) == "Undefined XRangeROI"

        roi.set_range(1, 2)
        assert str(roi) == "1.000 < x < 2.000"

    def test_reset(self):
        roi = XRangeROI()
        roi.set_range(1, 2)
        assert roi.defined()
        roi.reset()
        assert not roi.defined()

    def test_contains(self):
        roi = XRangeROI()
        roi.set_range(1, 3)
        x = np.array([0, 1, 2, 3])
        y = np.array([-np.inf, 100, 200, 0])
        np.testing.assert_array_equal(roi.contains(x, y),
                                      [False, False, True, False])

    def test_contains_undefined(self):
        roi = XRangeROI()
        with pytest.raises(UndefinedROI):
            roi.contains(1, 2)

    def test_to_polygon(self):
        roi = XRangeROI()
        assert roi.to_polygon() == ([], [])
        roi.set_range(1, 2)
        x, y = roi.to_polygon()
        np.testing.assert_array_equal(x, [1, 2, 2, 1, 1])
        np.testing.assert_array_equal(y,
                                      [-1e100, -1e100, 1e100, 1e100, -1e100])

    def test_serialization(self):
        roi = XRangeROI()
        roi.set_range(7, 8)
        new_roi = roundtrip_roi(roi)
        assert_almost_equal(new_roi.min, 7)
        assert_almost_equal(new_roi.max, 8)
        assert new_roi.ori == 'x'


class TestYRange(object):
    def test_undefined_on_init(self):
        assert not YRangeROI().defined()

    def test_str(self):
        roi = YRangeROI()
        assert str(roi) == "Undefined YRangeROI"

        roi.set_range(1, 2)
        assert str(roi) == "1.000 < y < 2.000"

    def test_reset(self):
        roi = YRangeROI()
        roi.set_range(1, 2)
        assert roi.defined()
        roi.reset()
        assert not roi.defined()

    def test_contains(self):
        roi = YRangeROI()
        roi.set_range(1, 3)
        y = np.array([0, 1, 2, 3])
        x = np.array([-np.inf, 100, 200, 0])
        np.testing.assert_array_equal(roi.contains(x, y),
                                      [False, False, True, False])

    def test_contains_undefined(self):
        roi = YRangeROI()
        with pytest.raises(UndefinedROI):
            roi.contains(1, 2)

    def test_to_polygon(self):
        roi = YRangeROI()
        assert roi.to_polygon() == ([], [])
        roi.set_range(1, 2)
        x, y = roi.to_polygon()
        np.testing.assert_array_equal(y, [1, 2, 2, 1, 1])
        np.testing.assert_array_equal(x,
                                      [-1e100, -1e100, 1e100, 1e100, -1e100])

    def test_serialization(self):
        roi = YRangeROI()
        roi.set_range(4, 5)
        new_roi = roundtrip_roi(roi)
        assert_almost_equal(new_roi.min, 4)
        assert_almost_equal(new_roi.max, 5)
        assert new_roi.ori == 'y'


class TestCircle(object):
    def setup_method(self, method):
        self.roi = CircularROI()

    def test_undefined_on_creation(self):
        assert not self.roi.defined()

    def test_contains_on_undefined_contains_raises(self):
        with pytest.raises(UndefinedROI):
            self.roi.contains(1, 1)

    def test_set_center(self):
        self.roi.move_to(0, 0)
        self.roi.set_radius(1)
        assert self.roi.contains(0, 0)
        assert not self.roi.contains(2, 2)
        self.roi.move_to(2, 2)
        assert not self.roi.contains(0, 0)
        assert self.roi.contains(2, 2)

    def test_set_radius(self):
        self.roi.move_to(0, 0)
        self.roi.set_radius(1)
        assert not self.roi.contains(1.5, 0)
        self.roi.set_radius(5)
        assert self.roi.contains(1.5, 0)

    def test_contains_many(self):
        x = [0, 0, 0, 0, 0]
        y = [0, 0, 0, 0, 0]
        self.roi.move_to(0, 0)
        self.roi.set_radius(1)
        assert all(self.roi.contains(x, y))
        assert all(self.roi.contains(np.asarray(x), np.asarray(y)))
        assert not any(self.roi.contains(np.asarray(x) + 10, y))

    def test_poly(self):
        self.roi.move_to(0, 0)
        self.roi.set_radius(1)
        x, y = self.roi.to_polygon()
        poly = PolygonalROI(vx=x, vy=y)
        assert poly.contains(0, 0)
        assert not poly.contains(10, 0)

    def test_poly_undefined(self):
        x, y = self.roi.to_polygon()
        assert x == []
        assert y == []

    def test_reset(self):
        assert not self.roi.defined()
        self.roi.move_to(0, 0)
        assert not self.roi.defined()
        self.roi.set_radius(2)
        assert self.roi.defined()
        self.roi.reset()
        assert not self.roi.defined()

    def test_multidim(self):
        self.roi.move_to(0, 0)
        self.roi.set_radius(1)
        x = np.array([.1, .2, .3, .4]).reshape(2, 2)
        y = np.array([-.1, -.2, -.3, -.4]).reshape(2, 2)
        assert self.roi.contains(x, y).all()
        assert not self.roi.contains(x + 1, y).any()
        assert self.roi.contains(x, y).shape == (2, 2)

    def test_serialization(self):
        self.roi.move_to(3, 4)
        self.roi.set_radius(5)
        new_roi = roundtrip_roi(self.roi)
        assert_almost_equal(new_roi.xc, 3)
        assert_almost_equal(new_roi.yc, 4)
        assert_almost_equal(new_roi.radius, 5)


class TestEllipse(object):

    def setup_method(self, method):
        self.roi_empty = EllipticalROI()
        self.roi = EllipticalROI(1, 2, 3, 4)
        self.roi_rotated = EllipticalROI(1, 2, 3, 0.4, theta=np.pi / 6)

    def test_undefined_on_creation(self):
        assert not self.roi_empty.defined()
        assert self.roi.defined()
        assert self.roi_rotated.defined()

    def test_contains_on_undefined_contains_raises(self):
        with pytest.raises(UndefinedROI):
            self.roi_empty.contains(1, 1)
        assert self.roi.contains(1, 1)
        assert self.roi_rotated.contains(2, 2.5)
        assert not self.roi_rotated.contains(2, 2)

    def test_set_center(self):
        assert self.roi.contains(0, 0)
        assert not self.roi.contains(12, 12)
        self.roi.xc = 11
        self.roi.yc = 12
        assert not self.roi.contains(0, 0)
        assert self.roi.contains(12, 12)

    def test_set_radius(self):
        assert self.roi.contains(0, 0)
        assert not self.roi.contains(12, 12)
        self.roi.radius_y = 100
        assert self.roi.contains(0, 0)
        assert not self.roi.contains(12, 12)
        self.roi.radius_x = 100
        assert self.roi.contains(0, 0)
        assert self.roi.contains(12, 12)

    def test_set_rotation(self):
        self.roi_rotated.rotate_to(0.55)
        assert self.roi_rotated.contains(-1.5, 0.4)
        self.roi_rotated.rotate_by(np.pi / 3)
        assert self.roi_rotated.contains(0.8, 4.5)
        self.roi_rotated.rotate_to(np.pi / 3)
        assert not self.roi_rotated.contains(0.8, 4.5)
        assert self.roi_rotated.contains(0.5, 0.5)

    def test_contains_many(self):
        x = [0, 0, 0, 0, 0]
        y = [0, 0, 0, 0, 0]
        assert all(self.roi.contains(x, y))
        assert all(self.roi.contains(np.asarray(x), np.asarray(y)))
        assert not any(self.roi.contains(np.asarray(x) + 10, y))

    def test_poly(self):
        x, y = self.roi.to_polygon()
        poly = PolygonalROI(vx=x, vy=y)
        assert poly.contains(0, 0)
        assert not poly.contains(10, 0)

        x, y = self.roi_rotated.to_polygon()
        poly = PolygonalROI(vx=x, vy=y)
        assert poly.contains(2, 2.5)
        assert not poly.contains(2, 2)

    def test_poly_undefined(self):
        x, y = self.roi_empty.to_polygon()
        assert x == []
        assert y == []

    def test_reset(self):
        assert not self.roi_empty.defined()
        self.roi_empty.xc = 1
        assert not self.roi_empty.defined()
        self.roi_empty.yc = 2
        assert not self.roi_empty.defined()
        self.roi_empty.radius_x = 3
        assert not self.roi_empty.defined()
        self.roi_empty.radius_y = 4
        assert self.roi_empty.defined()
        self.roi_empty.reset()
        assert not self.roi_empty.defined()

    def test_multidim(self):
        x = np.array([.1, .2, .3, .4]).reshape(2, 2)
        y = np.array([-.1, -.2, -.3, -.4]).reshape(2, 2)
        assert self.roi.contains(x, y).all()
        assert not self.roi.contains(x + 10, y).any()
        assert self.roi.contains(x, y).shape == (2, 2)

    def test_serialization(self):
        new_roi = roundtrip_roi(self.roi)
        assert_almost_equal(new_roi.xc, 1)
        assert_almost_equal(new_roi.yc, 2)
        assert_almost_equal(new_roi.radius_x, 3)
        assert_almost_equal(new_roi.radius_y, 4)

    def test_serialize_rotated(self):
        new_roi = roundtrip_roi(self.roi_rotated)
        assert new_roi.radius_y == 0.4
        assert new_roi.theta == np.pi / 6
        assert new_roi.contains(-1.5, 0.5)


class TestPolygon(object):
    def setup_method(self, method):
        self.roi = PolygonalROI()

    def define_as_square(self):
        self.roi.reset()
        assert not self.roi.defined()
        self.roi.add_point(0, 0)
        self.roi.add_point(0, 1)
        self.roi.add_point(1, 1)
        self.roi.add_point(1, 0)
        assert self.roi.defined()

    def test_contains_on_empty_raises(self):
        with pytest.raises(UndefinedROI):
            self.roi.contains(1, 2)

    def test_remove_empty(self):
        self.roi.remove_point(1, 0)

    def test_replace(self):
        self.define_as_square()
        assert self.roi.contains(.9, .02)
        self.roi.replace_last_point(0, 0)
        assert not self.roi.contains(.9, .01)

    def test_remove_successful(self):
        self.define_as_square()
        assert self.roi.contains(.9, .02)
        self.roi.remove_point(1, 0)
        assert not self.roi.contains(.9, .01)

    def test_nan(self):
        self.define_as_square()
        assert not self.roi.contains(np.nan, .5)

    def test_remove_unsuccessful(self):
        self.define_as_square()
        assert self.roi.contains(.9, .02)
        self.roi.remove_point(1.5, 0, thresh=.49)
        assert self.roi.contains(.9, .01)

    def test_to_poly(self):
        self.define_as_square()
        x, y = self.roi.to_polygon()
        assert x == [0, 0, 1, 1]
        assert y == [0, 1, 1, 0]

    def test_to_poly_undefined(self):
        assert self.roi.to_polygon() == ([], [])

    def test_rect(self):
        self.roi.reset()
        self.roi.add_point(4.95164474584, 0.136922625654)
        self.roi.add_point(6.08256580223, 0.136922625654)
        self.roi.add_point(6.08256580223, 0.423771697842)
        self.roi.add_point(4.95164474584, 0.423771697842)
        self.roi.add_point(4.95164474584, 0.136922625654)
        x = np.array([4.4, 3.18, 4.49, 4.49])
        y = np.array([.2, .2, .2, .2])
        assert not self.roi.contains(4.4000001, 0.2)
        assert not self.roi.contains(3.1800001, 0.2)
        assert not self.roi.contains(4.4899998, 0.2)
        assert not self.roi.contains(x, y).any()
        x.shape = (2, 2)
        y.shape = (2, 2)
        assert not self.roi.contains(x, y).any()
        assert self.roi.contains(x, y).shape == (2, 2)

    def test_empty(self):
        assert not self.roi.defined()
        with pytest.raises(UndefinedROI):
            self.roi.contains(0, 0)

    def test_contains_scalar(self):
        self.define_as_square()
        assert self.roi.contains(.5, .5)
        assert not self.roi.contains(1.5, 1.5)

    def test_contains_list(self):
        self.define_as_square()
        assert self.roi.contains([.5, .4], [.5, .4]).all()
        assert not self.roi.contains([1.5, 1.5], [0, 0]).any()

    def test_contains_numpy(self):
        self.define_as_square()
        x = np.array([.4, .5, .4])
        y = np.array([.4, .5, .4])
        assert self.roi.contains(x, y).all()
        assert not self.roi.contains(x + 5, y).any()

    def test_str(self):
        """ __str__ returns a string """
        assert type(str(self.roi)) == str

    def test_rotate(self):
        """ Test 45 deg rotation of square ROI """
        self.define_as_square()
        self.roi.rotate_to(np.pi/4)
        assert self.roi.contains([.5, .5, 1.2], [1.2, -0.2, .5]).all()
        assert not self.roi.contains([1.5, 1.5], [0, 0]).any()
        assert_almost_equal(self.roi.centroid(), (0.5, 0.5), decimal=12)

    def test_rotate_triangle(self):
        """ Test incremental rotations of triangular (half-square) ROI """
        self.define_as_square()
        assert_almost_equal(self.roi.area(), 1, decimal=12)
        assert_almost_equal(self.roi.centroid(), (0.5, 0.5), decimal=12)
        self.roi.remove_point(1, 1)
        assert_almost_equal(self.roi.area(), 0.5, decimal=12)
        assert_almost_equal(self.roi.centroid(), (1/3, 1/3), decimal=12)
        self.roi.rotate_to(np.pi/3)
        assert_almost_equal(self.roi.centroid(), (1/3, 1/3), decimal=12)
        self.roi.rotate_by(np.pi/6)
        assert_almost_equal(self.roi.area(), 0.5, decimal=12)
        assert_almost_equal(self.roi.centroid(), (1/3, 1/3), decimal=12)
        assert_almost_equal(self.roi.vx, (2/3, -1/3, 2/3), decimal=12)
        assert_almost_equal(self.roi.vy, (0, 0, 1), decimal=12)

    def test_rotate_polyline(self):
        """ Test rotation of degenerate (linear) ROI around mean """
        self.roi.reset()
        self.roi.add_point(-2, 0)
        self.roi.add_point(4, 0)
        assert_almost_equal(self.roi.mean(), (1.0, 0.0), decimal=12)
        self.roi.add_point(-0.5, 0)
        self.roi.add_point(-1.5, 0)
        assert_almost_equal(self.roi.mean(), (0.0, 0.0), decimal=12)
        assert all(np.isnan(self.roi.centroid()))
        self.roi.rotate_to(np.pi/2)
        assert_almost_equal(self.roi.vx, (0, 0, 0, 0), decimal=12)
        assert_almost_equal(self.roi.vy, (-2, 4, -0.5, -1.5), decimal=12)

    def test_append_mock_points(self):
        """
        Test that adding points on the side of square ROI conserves area and centroid.
        """
        self.define_as_square()
        assert_almost_equal(self.roi.area(), 1, decimal=12)
        assert_almost_equal(self.roi.area(signed=True), -1, decimal=12)
        assert_almost_equal(self.roi.centroid(), (0.5, 0.5), decimal=12)
        assert_almost_equal(self.roi.mean(), (0.5, 0.5), decimal=12)
        self.roi.add_point(0.9, 0)
        self.roi.add_point(0.7, 0)
        assert_almost_equal(self.roi.area(), 1, decimal=12)
        assert_almost_equal(self.roi.centroid(), (0.5, 0.5), decimal=12)
        assert_almost_equal(self.roi.mean(), (0.6, 1/3), decimal=12)

    def test_serialization(self):
        self.define_as_square()
        new_roi = roundtrip_roi(self.roi)
        assert_almost_equal(new_roi.vx, np.array([0, 0, 1, 1]))
        assert_almost_equal(new_roi.vy, np.array([0, 1, 1, 0]))
        self.roi.rotate_to(np.pi / 4)
        new_roi = roundtrip_roi(self.roi)
        assert new_roi.theta == 0
        sqh = np.sqrt(0.5)
        assert_almost_equal(new_roi.vx, np.array([0, -sqh, 0, sqh]) + 0.5)
        assert_almost_equal(new_roi.vy, np.array([-sqh, 0, sqh, 0]) + 0.5)


class TestProjected3dROI(object):
    # matrix that converts xyzw to yxzw
    xyzw2yxzw = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    x = [1, 2, 3]
    y = [2, 3, 4]
    z = [5, 6, 7]
    # repeat the arrays, 'rolled over' by 1
    x_nd = [[1, 3], [2, 1], [3, 2]]
    y_nd = [[2, 3], [3, 2], [4, 3]]
    z_nd = [[5, 7], [6, 5], [7, 6]]

    def test_contains2d(self):
        roi_2d = PolygonalROI(vx=[0.5, 2.5, 2.5, 0.5], vy=[1, 1, 3.5, 3.5])
        roi = Projected3dROI(roi_2d=roi_2d, projection_matrix=np.eye(4))
        assert roi.contains(self.x, self.y).tolist() == [True, True, False]

    def test_contains3d(self):
        roi_2d = PolygonalROI(vx=[1.5, 3.5, 3.5, 1.5], vy=[4, 4, 6.5, 6.5])
        roi = Projected3dROI(roi_2d=roi_2d, projection_matrix=self.xyzw2yxzw)
        assert roi.contains3d(self.x, self.y, self.z).tolist() == [True, True, False]
        assert roi.contains3d(self.x_nd, self.y_nd, self.z_nd).tolist() == [[True, False], [True, True], [False, True]]

    def test_forward(self):
        # testing the calls that should be forwarded to roi_2d
        roi_2d = PolygonalROI(vx=[0.5, 2.5, 2.5, 0.5], vy=[1, 1, 3.5, 3.5])
        roi = Projected3dROI(roi_2d=roi_2d, projection_matrix=np.eye(4))

        assert roi.contains(self.x, self.y).tolist() == roi_2d.contains(self.x, self.y).tolist()
        assert roi.to_polygon() == roi_2d.to_polygon()
        assert roi.defined() == roi_2d.defined()

    def test_rotate2d(self):
        """ Test rotation of the 2d ROI """
        roi_2d = PolygonalROI(vx=[1.5, 3.5, 3.5, 1.5], vy=[4.5, 4.5, 5.5, 5.5])
        roi = Projected3dROI(roi_2d=roi_2d, projection_matrix=self.xyzw2yxzw)
        assert roi.contains3d(self.x, self.y, self.z).tolist() == [True, False, False]
        assert roi.contains3d(self.x_nd, self.y_nd, self.z_nd).tolist() == [[True, False], [False, True], [False, False]]

        roi.rotate_by(np.pi / 2)
        assert_almost_equal(roi.roi_2d.theta, np.pi / 2)
        assert_almost_equal(roi_2d.theta, np.pi / 2)
        assert roi.contains3d(self.x, self.y, self.z).tolist() == [True, True, False]
        assert roi.contains3d(self.x_nd, self.y_nd, self.z_nd).tolist() == [[True, False], [True, True], [False, True]]


class TestCategorical(object):

    def test_empty(self):

        roi = CategoricalROI()
        assert not roi.defined()

    def test_defined(self):

        nroi = CategoricalROI(categories=['a', 'b', 'c'])
        assert nroi.defined()
        nroi.reset()
        assert not nroi.defined()

    def test_loads_from_components(self):

        roi = CategoricalROI()
        comp = CategoricalComponent(np.array(['a', 'a', 'b']))
        roi.update_categories(comp)

        np.testing.assert_array_equal(roi.categories,
                                      np.array(['a', 'b']))

        roi = CategoricalROI(categories=comp)
        np.testing.assert_array_equal(roi.categories,
                                      np.array(['a', 'b']))

    def test_applies_components(self):

        roi = CategoricalROI()
        comp = CategoricalComponent(np.array(['a', 'b', 'c']))
        roi.update_categories(CategoricalComponent(np.array(['a', 'b'])))
        contained = roi.contains(comp, None)
        np.testing.assert_array_equal(contained,
                                      np.array([True, True, False]))

    def test_from_range(self):

        comp = CategoricalComponent(np.array(list('abcdefghijklmnopqrstuvwxyz') * 2))

        roi = CategoricalROI.from_range(comp, 6, 10)
        np.testing.assert_array_equal(roi.categories,
                                      np.array(list('ghij')))

    def test_empty_categories(self):
        roi = CategoricalROI()
        contains = roi.contains(np.array(['a', 'b', 'c']), None)
        np.testing.assert_array_equal(contains, [0, 0, 0])


class DummyEvent(object):
    def __init__(self, x, y, inaxes=True, key=None):
        self.inaxes = inaxes
        self.xdata = x
        self.ydata = y
        self.key = key


class MockAxes(object):
    def __init__(self):
        self.figure = MagicMock()
        self.figure.canvas = MagicMock()

    def add_patch(self, patch):
        pass


class TestMpl(object):

    def setup_method(self, method):
        self.axes = MagicMock()
        self.roi = self._roi_factory()

    def _roi_factory(self):
        raise NotImplementedError

    def test_undefined_on_creation(self):
        assert not self.roi._roi.defined()

    def test_large_zorder(self):
        assert self.roi._patch.get_zorder() >= 100

    def test_proper_roi(self):
        raise NotImplementedError

    def test_start_ignored_if_not_inaxes(self):
        ev = DummyEvent(0, 0, inaxes=None)
        self.roi.start_selection(ev)
        assert not self.roi._roi.defined()

    def test_canvas_syncs_properly(self):
        assert self.axes.figure.canvas.draw_idle.call_count == 0
        assert self.axes.figure.canvas.blit.call_count == 0
        event = DummyEvent(5, 5, inaxes=self.axes)
        self.roi.start_selection(event)
        assert self.axes.figure.canvas.draw_idle.call_count == 0
        assert self.axes.figure.canvas.blit.call_count == 1
        self.roi.update_selection(event)
        assert self.axes.figure.canvas.draw_idle.call_count == 0
        assert self.axes.figure.canvas.blit.call_count == 2
        self.roi.update_selection(event)
        assert self.axes.figure.canvas.draw_idle.call_count == 0
        assert self.axes.figure.canvas.blit.call_count == 3
        self.roi.finalize_selection(event)
        assert self.axes.figure.canvas.draw_idle.call_count == 0
        assert self.axes.figure.canvas.blit.call_count == 4

    def test_patch_shown_on_start(self):
        assert not self.roi._patch.get_visible()
        event = DummyEvent(5, 5, inaxes=self.axes)
        self.roi.start_selection(event)
        assert self.roi._patch.get_visible()

    def test_patch_hidden_on_finalise(self):
        event = DummyEvent(5, 5, inaxes=self.axes)
        self.roi.start_selection(event)
        self.roi.finalize_selection(event)
        assert not self.roi._patch.get_visible()

    def test_update_before_start_ignored(self):
        self.roi.update_selection(None)
        assert not self.roi._roi.defined()

    def test_finalize_before_start_ignored(self):
        self.roi.finalize_selection(None)
        assert not self.roi._roi.defined()

    def test_roi_defined_after_start(self):
        event = DummyEvent(5, 5, inaxes=self.axes)
        self.roi.start_selection(event)
        assert self.roi._roi.defined()

    def test_roi_undefined_before_start(self):
        assert not self.roi._roi.defined()

    def scrub(self, roi=None, abort=False, outside=False):

        if roi is None:

            roi = self._roi_factory()

            event = DummyEvent(5, 5, inaxes=self.axes)
            roi.start_selection(event)
            event = DummyEvent(10, 10, inaxes=self.axes)
            roi.update_selection(event)
            roi.finalize_selection(DummyEvent(10, 10))

        if outside:
            roi.start_selection(DummyEvent(16, 16, inaxes=self.axes, key='control'))
            roi.update_selection(DummyEvent(17, 18, inaxes=self.axes, key='control'))
        else:
            roi.start_selection(DummyEvent(6, 6, inaxes=self.axes, key='control'))
            roi.update_selection(DummyEvent(7, 8, inaxes=self.axes, key='control'))

        if abort:
            roi.abort_selection(None)

        return roi


class TestRectangleMpl(TestMpl):

    def _roi_factory(self):
        return MplRectangularROI(self.axes)

    def test_scrub(self):

        roi = self.scrub()

        assert roi._roi.xmin == 6
        assert roi._roi.xmax == 11
        assert roi._roi.ymin == 7
        assert roi._roi.ymax == 12

    def test_abort(self):

        roi = self.scrub(abort=True)

        assert roi._roi.xmin == 5
        assert roi._roi.xmax == 10
        assert roi._roi.ymin == 5
        assert roi._roi.ymax == 10

    def test_outside(self):

        roi = self.scrub(outside=True)

        assert roi._roi.xmin == 5
        assert roi._roi.xmax == 10
        assert roi._roi.ymin == 5
        assert roi._roi.ymax == 10

    def assert_roi_correct(self, x0, x1, y0, y1):
        corner = self.roi.roi().corner()
        height = self.roi.roi().height()
        width = self.roi.roi().width()
        assert self.roi.roi().defined()
        assert_almost_equal(corner[0], min(x0, x1), 4)
        assert_almost_equal(corner[1], min(y0, y1), 4)
        assert_almost_equal(abs(y1 - y0), height, 4)
        assert_almost_equal(abs(x1 - x0), width, 4)

    def assert_patch_correct(self, x0, x1, y0, y1):
        corner = self.roi._patch.get_xy()
        width = self.roi._patch.get_width()
        height = self.roi._patch.get_height()
        assert_almost_equal(corner, (min(x0, x1), min(y0, y1)), 4)
        assert_almost_equal(width, abs(x0 - x1))
        assert_almost_equal(height, abs(y0 - y1))

    def test_str(self):
        assert type(str(self.roi)) == str

    def test_proper_roi(self):
        assert isinstance(self.roi._roi, RectangularROI)

    def test_roi_on_start_selection(self):
        event = DummyEvent(5, 5, inaxes=self.axes)
        self.roi.start_selection(event)
        self.assert_roi_correct(5, 5, 5, 5)

    def test_patch_on_start_selection(self):
        event = DummyEvent(5, 5, inaxes=self.axes)
        self.roi.start_selection(event)
        self.assert_patch_correct(5, 5, 5, 5)

    def test_update_selection(self):
        event = DummyEvent(5, 6, inaxes=self.axes)
        self.roi.start_selection(event)
        event = DummyEvent(10, 11, inaxes=self.axes)
        self.roi.update_selection(event)
        self.assert_roi_correct(5, 10, 6, 11)
        self.assert_patch_correct(5, 10, 6, 11)

    def test_finalize_selection(self):
        event = DummyEvent(5, 6, inaxes=self.axes)
        self.roi.start_selection(event)
        event = DummyEvent(10, 11, inaxes=self.axes)
        self.roi.update_selection(event)
        self.roi.finalize_selection(event)
        self.assert_roi_correct(5, 10, 6, 11)
        self.assert_patch_correct(5, 10, 6, 11)

    def test_define_roi_to_right(self):
        ev0 = DummyEvent(0.5, 0.5, inaxes=self.axes)
        ev1 = DummyEvent(1, 1, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)

        self.assert_roi_correct(.5, 1, .5, 1)
        self.assert_patch_correct(.5, 1, .5, 1)

    def test_define_roi_to_left(self):
        ev0 = DummyEvent(1, 1, inaxes=self.axes)
        ev1 = DummyEvent(0.5, 0.5, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)

        self.assert_roi_correct(.5, 1, .5, 1)
        self.assert_patch_correct(.5, 1, .5, 1)


class TestXRangeMpl(TestMpl):
    def _roi_factory(self):
        return MplXRangeROI(self.axes)

    def test_proper_roi(self):
        assert isinstance(self.roi._roi, XRangeROI)

    def test_start_selection(self):
        event = DummyEvent(1, 1, inaxes=self.axes)
        self.roi.start_selection(event)
        assert self.roi._roi.defined()

    def test_update_selection(self):
        ev0 = DummyEvent(1, 1, inaxes=self.axes)
        ev1 = DummyEvent(2, 1, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        assert self.roi._roi.defined()
        assert self.roi._roi.range() == (1, 2)

    def test_finalize_selection(self):
        ev0 = DummyEvent(1, 1, inaxes=self.axes)
        ev1 = DummyEvent(2, 1, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)
        assert self.roi._roi.defined()
        assert self.roi._roi.range() == (1, 2)

        assert not self.roi._patch.get_visible()

    def test_scrub(self):

        roi = self.scrub()

        assert_almost_equal(roi._roi.min, 3.0)
        assert_almost_equal(roi._roi.max, 8.0)

    def test_abort(self):

        roi = self.scrub(abort=True)

        assert_almost_equal(roi._roi.min, 5.0)
        assert_almost_equal(roi._roi.max, 10.0)

    def test_outside(self):

        roi = self.scrub(outside=True)

        assert_almost_equal(roi._roi.min, 5.0)
        assert_almost_equal(roi._roi.max, 10.0)


class TestYRangeMpl(TestMpl):
    def _roi_factory(self):
        return MplYRangeROI(self.axes)

    def test_proper_roi(self):
        assert isinstance(self.roi._roi, YRangeROI)

    def test_start_selection(self):
        event = DummyEvent(1, 1, inaxes=self.axes)
        self.roi.start_selection(event)
        assert self.roi._roi.defined()

    def test_update_selection(self):
        ev0 = DummyEvent(1, 1, inaxes=self.axes)
        ev1 = DummyEvent(1, 2, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        assert self.roi._roi.defined()
        assert self.roi._roi.range() == (1, 2)

    def test_finalize_selection(self):
        ev0 = DummyEvent(1, 1, inaxes=self.axes)
        ev1 = DummyEvent(1, 2, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)
        assert self.roi._roi.defined()
        assert self.roi._roi.range() == (1, 2)

        assert not self.roi._patch.get_visible()

    def test_scrub(self):

        roi = self.scrub()

        assert_almost_equal(roi._roi.min, 4.0)
        assert_almost_equal(roi._roi.max, 9.0)

    def test_abort(self):

        roi = self.scrub(abort=True)

        assert_almost_equal(roi._roi.min, 5.0)
        assert_almost_equal(roi._roi.max, 10.0)

    def test_outside(self):

        roi = self.scrub(outside=True)

        assert_almost_equal(roi._roi.min, 5.0)
        assert_almost_equal(roi._roi.max, 10.0)


def test_circular_roi_representation():

    # Test cases where drawn circular ROIs are converted to circles, ellipses,
    # and polygons. Here we don't override pixel_to_data and data_to_pixel
    # since these are important.

    event = namedtuple('Event', ['inaxes', 'xdata', 'ydata', 'key'])

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    # Case 1: log-linear axes

    ax.set_xlim(10, 100)
    ax.set_ylim(-2, 5)
    ax.set_xscale('log')
    ax.set_yscale('linear')

    fig.canvas.draw()

    mpl_roi = MplCircularROI(ax)
    mpl_roi.start_selection(event(inaxes=ax, xdata=50, ydata=1, key=None))
    mpl_roi.update_selection(event(inaxes=ax, xdata=100, ydata=1, key=None))
    roi = mpl_roi.roi()
    assert isinstance(roi, PolygonalROI)

    # Case 2: linear-linear axes with non-square aspect ratio

    ax.set_xlim(10, 100)
    ax.set_ylim(-2, 5)
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    fig.canvas.draw()

    mpl_roi = MplCircularROI(ax)
    mpl_roi.start_selection(event(inaxes=ax, xdata=30, ydata=1, key=None))
    mpl_roi.update_selection(event(inaxes=ax, xdata=80, ydata=1, key=None))
    roi = mpl_roi.roi()
    assert isinstance(roi, EllipticalROI)
    assert_allclose(roi.xc, 30)
    assert_allclose(roi.yc, 1)
    assert_allclose(roi.radius_x, 50)
    assert_allclose(roi.radius_y, 5.871212)

    # Case 3: linear-linear axes with square aspect ratio

    ax.set_aspect('equal')
    ax.set_xlim(-40, 100)

    fig.canvas.draw()

    mpl_roi = MplCircularROI(ax)
    mpl_roi.start_selection(event(inaxes=ax, xdata=50, ydata=1, key=None))
    mpl_roi.update_selection(event(inaxes=ax, xdata=80, ydata=1, key=None))
    roi = mpl_roi.roi()
    assert isinstance(roi, CircularROI)
    assert_allclose(roi.xc, 50)
    assert_allclose(roi.yc, 1)
    assert_allclose(roi.radius, 30)


class TestCircleMpl(TestMpl):

    def _roi_factory(self):
        return MplCircularROI(self.axes)

    def setup_method(self, method):
        super(TestCircleMpl, self).setup_method(method)
        self.pixel_to_data = r.pixel_to_data
        self.data_to_pixel = r.data_to_pixel

        r.pixel_to_data = lambda x, y, z: np.column_stack((y, z))
        r.data_to_pixel = lambda x, y, z: np.column_stack((y, z))

    def teardown_method(self, method):
        # restore methods
        r.pixel_to_data = self.pixel_to_data
        r.data_to_pixel = self.data_to_pixel

    def test_proper_roi(self):
        assert isinstance(self.roi._roi, CircularROI)

    def test_to_polygon_undefined(self):
        """to_polygon() result should be undefined before defining polygon"""
        roi = self.roi.roi()
        assert not roi.defined()

    def test_roi_defined_correctly(self):
        ev0 = DummyEvent(0, 0, inaxes=self.axes)
        ev1 = DummyEvent(5, 0, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.finalize_selection(ev1)
        self.assert_roi_correct(0, 0, 5)

    def assert_roi_correct(self, x, y, r):
        roi = self.roi.roi()
        assert roi.defined()
        assert roi.contains(x, y)
        assert roi.contains(x + .95 * r, y)
        assert not roi.contains(x + 1.05 * r, y)
        assert not roi.contains(x + .8 * r, y + .8 * r)

    def test_scrub(self):

        roi = self.scrub()

        assert roi._roi.xc == 6
        assert roi._roi.yc == 7
        assert_almost_equal(roi._roi.radius, 7.0, decimal=0)

    def test_abort(self):

        roi = self.scrub(abort=True)

        assert roi._roi.xc == 5
        assert roi._roi.yc == 5
        assert_almost_equal(roi._roi.radius, 7.0, decimal=0)

    def test_outside(self):

        roi = self.scrub(outside=True)

        assert roi._roi.xc == 5
        assert roi._roi.yc == 5
        assert_almost_equal(roi._roi.radius, 7.0, decimal=0)


class TestPolyMpl(TestMpl):
    def _roi_factory(self):
        return MplPolygonalROI(self.axes)

    def test_proper_roi(self):
        return isinstance(self.roi._roi, PolygonalROI)

    def send_events(self):
        ev0 = DummyEvent(5, 5, inaxes=self.axes)
        ev1 = DummyEvent(5, 10, inaxes=self.axes)
        ev2 = DummyEvent(10, 10, inaxes=self.axes)
        ev3 = DummyEvent(10, 5, inaxes=self.axes)
        self.roi.start_selection(ev0)
        self.roi.update_selection(ev1)
        self.roi.update_selection(ev2)
        self.roi.update_selection(ev3)
        self.roi.finalize_selection(ev3)

    def assert_roi_correct(self):
        roi = self.roi.roi()
        assert roi.contains(7.0, 7.0)
        assert not roi.contains(12, 7.0)

    def test_define(self):
        self.send_events()
        self.assert_roi_correct()

    def test_scrub(self):

        self.send_events()

        roi = self.scrub(roi=self.roi)

        assert roi._roi.vx[0] == 6
        assert roi._roi.vy[0] == 7

    def test_abort(self):

        self.send_events()

        roi = self.scrub(roi=self.roi, abort=True)

        assert roi._roi.vx[0] == 5
        assert roi._roi.vy[0] == 5

    def test_outside(self):

        self.send_events()

        roi = self.scrub(roi=self.roi, outside=True)

        assert roi._roi.vx[0] == 5
        assert roi._roi.vy[0] == 5


class TestPickMpl(TestMpl):
    def _roi_factory(self):
        return MplPickROI(self.axes)

    def test_proper_roi(self):
        return isinstance(self.roi._roi, PointROI)

    def test_start_ignored_if_not_inaxes(self):
        ev = DummyEvent(0, 0, inaxes=None)
        self.roi.start_selection(ev)
        assert self.roi._roi.defined()

    def test_update_before_start_ignored(self):
        ev = DummyEvent(0, 0, inaxes=None)
        self.roi.update_selection(ev)
        assert self.roi._roi.defined()

    def test_finalize_before_start_ignored(self):
        ev = DummyEvent(0, 0, inaxes=None)
        self.roi.finalize_selection(ev)
        assert self.roi._roi.defined()

    def test_large_zorder(self):
        """No patch to test for."""

    def test_patch_shown_on_start(self):
        """No patch to test for."""

    def test_patch_hidden_on_finalise(self):
        """No patch to test for."""

    def test_canvas_syncs_properly(self):
        """No patch to test for."""


class TestUtil(object):

    def setup_method(self, method):
        self.axes = AXES

    def test_aspect_ratio(self):
        self.axes.figure.set_figheight(5)
        self.axes.figure.set_figwidth(5)
        self.axes.set_ylim([0, 10])
        self.axes.set_xlim([0, 10])

        ax0 = r.aspect_ratio(self.axes)
        self.axes.set_ylim(0, 20)
        assert_almost_equal(r.aspect_ratio(self.axes), ax0 / 2, 4)
        self.axes.set_ylim(0, 5)
        assert_almost_equal(r.aspect_ratio(self.axes), ax0 * 2, 4)
        self.axes.set_xlim(0, 5)
        assert_almost_equal(r.aspect_ratio(self.axes), ax0, 4)
        self.axes.set_xlim(0, 10)
        assert_almost_equal(r.aspect_ratio(self.axes), ax0 * 2, 4)

    def test_data_to_norm_with_scalars(self):
        # assume data that gets submitted to axes is valid.
        # testing to see if we can get there
        self.axes.set_xlim(0, 10)
        self.axes.set_ylim(0, 10)
        func = r.data_to_norm
        assert_almost_equal(func(self.axes, 0, 0)[0, 0], 0, 3)
        assert_almost_equal(func(self.axes, 0, 0)[0, 1], 0, 3)
        assert_almost_equal(func(self.axes, 0, 10)[0, 0], 0, 3)
        assert_almost_equal(func(self.axes, 0, 10)[0, 1], 1, 3)
        assert_almost_equal(func(self.axes, 10, 10)[0, 0], 1, 3)
        assert_almost_equal(func(self.axes, 10, 10)[0, 1], 1, 3)
        assert_almost_equal(func(self.axes, 10, 0)[0, 0], 1, 3)
        assert_almost_equal(func(self.axes, 10, 0)[0, 1], 0, 3)

        x = np.array([0, 0, 10, 10])
        y = np.array([0, 10, 0, 10])
        xans = [0, 0, 1, 1]
        yans = [0, 1, 0, 1]
        norm = func(self.axes, x, y)
        assert_almost_equal(norm[0, 0], xans[0], 3)
        assert_almost_equal(norm[1, 0], xans[1], 3)
        assert_almost_equal(norm[2, 0], xans[2], 3)
        assert_almost_equal(norm[3, 0], xans[3], 3)
        assert_almost_equal(norm[0, 1], yans[0], 3)
        assert_almost_equal(norm[1, 1], yans[1], 3)
        assert_almost_equal(norm[2, 1], yans[2], 3)
        assert_almost_equal(norm[3, 1], yans[3], 3)

        x = [0, 0, 10, 10]
        y = [0, 10, 0, 10]
        assert_almost_equal(norm[0, 0], xans[0], 3)
        assert_almost_equal(norm[1, 0], xans[1], 3)
        assert_almost_equal(norm[2, 0], xans[2], 3)
        assert_almost_equal(norm[3, 0], xans[3], 3)
        assert_almost_equal(norm[0, 1], yans[0], 3)
        assert_almost_equal(norm[1, 1], yans[1], 3)
        assert_almost_equal(norm[2, 1], yans[2], 3)
        assert_almost_equal(norm[3, 1], yans[3], 3)

    def test_data_to_pixel(self):
        xp = 100
        yp = 100
        data = r.pixel_to_data(self.axes, xp, yp)
        pixel = r.data_to_pixel(self.axes, data[:, 0], data[:, 1])
        assert_almost_equal(pixel[0, 0], xp, 3)
        assert_almost_equal(pixel[0, 1], yp, 3)


del TestMpl  # prevents unittest discovery from running abstract base class
