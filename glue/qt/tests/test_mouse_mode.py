from mock import MagicMock, patch
import PyQt4
import numpy as np
from numpy.testing import assert_almost_equal

from ..mouse_mode import MouseMode, RectangleMode, CircleMode, PolyMode, \
                         ContrastMode, ContourMode, contour_to_roi

class Event(object):
    def __init__(self, x, y, button=3):
        self.x = x
        self.y = y
        self.xdata = x
        self.ydata = y
        self.button = button


def axes():
    result = MagicMock()
    result.figure.canvas.get_width_height.return_value = (640, 480)
    return result


class TestMouseMode(object):

    @classmethod
    def setup_class(cls):
        cls.app = PyQt4.QtGui.QApplication([''])

    def setup_method(self, method):
        self.mode = self.mode_factory()(axes())
        self.axes = self.mode._axes
        self.attach_callbacks()

    def attach_callbacks(self):
        self.press = self.mode._press_callback = MagicMock()
        self.move = self.mode._move_callback = MagicMock()
        self.release = self.mode._release_callback = MagicMock()

    def mode_factory(self):
        return MouseMode

    def test_press_callback(self):
        e = Event(1, 2)
        self.mode.press(e)
        self.press.assert_called_once_with(self.mode)
        assert self.move.call_count == 0
        assert self.release.call_count == 0

    #def test_log_null_event(self):
    #    """ Should exit quietly if event is None """
    #    self.mode._log_position(None)

    def test_move_callback(self):
        e = Event(1, 2)
        self.mode.move(e)
        self.move.assert_called_once_with(self.mode)
        assert self.press.call_count == 0
        assert self.release.call_count == 0

    def test_release_callback(self):
        e = Event(1, 2)
        self.mode.release(e)
        self.release.assert_called_once_with(self.mode)
        assert self.press.call_count == 0
        assert self.move.call_count == 0

    def test_press_log(self):
        e = Event(1, 2)
        self.mode.press(e)
        assert self.mode._event_x == 1
        assert self.mode._event_y == 2

    def test_move_log(self):
        e = Event(1, 2)
        self.mode.move(e)
        assert self.mode._event_x == 1
        assert self.mode._event_y == 2

    def test_release_log(self):
        e = Event(1, 2)
        self.mode.release(e)
        assert self.mode._event_x == 1
        assert self.mode._event_y == 2


class TestRoiMode(TestMouseMode):

    def setup_method(self, method):
        TestMouseMode.setup_method(self, method)
        self.mode._roi_tool = MagicMock()

    def mode_factory(self):
        raise NotImplemented

    def test_roi_called_on_press(self):
        e = Event(1, 2)
        self.mode.press(e)
        self.mode._roi_tool.start_selection.assert_called_once_with(e)

    def test_roi_called_on_move(self):
        e = Event(1, 2)
        self.mode.move(e)
        self.mode._roi_tool.update_selection.assert_called_once_with(e)

    def test_roi_called_on_release(self):
        e = Event(1, 2)
        self.mode.release(e)
        self.mode._roi_tool.finalize_selection.assert_called_once_with(e)

    def test_roi(self):
        r = self.mode.roi()
        self.mode._roi_tool.roi.assert_called_once_with()


class TestRectangleMode(TestRoiMode):
    def mode_factory(self):
        return RectangleMode


class TestCircleMode(TestRoiMode):
    def mode_factory(self):
        return CircleMode


class TestPolyMode(TestRoiMode):
    def mode_factory(self):
        return PolyMode


class TestContrastMode(TestMouseMode):
    def mode_factory(self):
        return ContrastMode

    def test_move_ignored_if_not_right_drag(self):
        e = Event(1, 2, button=1)
        self.mode.move(e)
        count = self.mode._axes.figure.canvas.get_width_height.call_count
        assert count == 0

    def test_get_scaling(self):
        data = [1, 2, 3]
        self.mode.bias = 0.
        self.mode.contrast = 2.
        lo, hi = self.mode.get_scaling(data)
        assert_almost_equal(lo, -3)
        assert_almost_equal(hi, 5)


class TestContourMode(TestMouseMode):

    def mode_factory(self):
        return ContourMode

    def test_roi_before_event(self):
        data = MagicMock()
        roi = self.mode.roi(data)
        assert roi is None

    def test_roi(self):
        with patch('glue.qt.mouse_mode.contour_to_roi') as cntr:
            data = MagicMock()
            e = Event(1, 2)
            self.mode.press(e)
            self.mode.roi(data)
            cntr.assert_called_once_with(1, 2, data)


class TestContourToRoi(object):

    def test_roi(self):
        with patch('glue.core.util.point_contour') as point_contour:
            point_contour.return_value = np.array([[1, 2], [2, 3]])
            p = contour_to_roi(1, 2, None)
            np.testing.assert_array_almost_equal(p.vx, [1, 2])
            np.testing.assert_array_almost_equal(p.vy, [2, 3])

    def test_roi_null_result(self):
        with patch('glue.core.util.point_contour') as point_contour:
            point_contour.return_value = None
            p = contour_to_roi(1, 2, None)
            assert p is None

del TestRoiMode  # prevents test discovery from running abstract test
