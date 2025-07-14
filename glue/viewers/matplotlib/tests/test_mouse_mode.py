# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from unittest.mock import MagicMock

# from glue.utils.qt import process_dialog

from ..mouse_mode import MouseMode
from ..toolbar_mode import RectangleMode, CircleMode, PolyMode


class Event(object):

    def __init__(self, x, y, button=3, key='a'):
        self.x = x
        self.y = y
        self.xdata = x
        self.ydata = y
        self.button = button
        self.key = key
        self.inaxes = True


def viewer():
    result = MagicMock()
    result.axes.figure.canvas.get_width_height.return_value = (640, 480)
    return result


class TestMouseMode(object):

    def setup_method(self, method):
        self.mode = self.mode_factory()(viewer())
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

    # def test_log_null_event(self):
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
        raise NotImplementedError()

    def test_roi_not_called_on_press(self):
        e = Event(1, 2)
        self.mode.press(e)
        assert self.mode._roi_tool.start_selection.call_count == 0

    def test_roi_called_on_drag(self):
        e = Event(1, 2)
        e2 = Event(10, 200)
        self.mode.press(e)
        self.mode.move(e2)
        self.mode._roi_tool.start_selection.assert_called_once_with(e)
        self.mode._roi_tool.update_selection.assert_called_once_with(e2)

    # def test_roi_ignores_small_drags(self):
    #     e = Event(1, 2)
    #     e2 = Event(1, 3)
    #     self.mode.press(e)
    #     self.mode.move(e2)
    #     assert self.mode._roi_tool.start_selection.call_count == 0
    #     assert self.mode._roi_tool.update_selection.call_count == 0

    def test_roi_called_on_release(self):
        e = Event(1, 2)
        e2 = Event(10, 20)
        self.mode.press(e)
        self.mode.move(e2)
        self.mode.release(e2)
        self.mode._roi_tool.finalize_selection.assert_called_once_with(e2)

    def test_roi(self):
        self.mode.roi()
        self.mode._roi_tool.roi.assert_called_once_with()

    def test_roi_resets_on_escape(self):
        e = Event(1, 2)
        e2 = Event(1, 30, key='escape')
        self.mode.press(e)
        self.mode.key(e2)
        self.mode.press(e)
        assert self.mode._roi_tool.abort_selection.call_count == 1


class TestClickRoiMode(TestMouseMode):

    def setup_method(self, method):
        TestMouseMode.setup_method(self, method)
        self.mode._roi_tool = MagicMock()
        self.mode._roi_tool.active.return_value = False

    def mode_factory(self):
        raise NotImplementedError()

    def test_roi_started_on_press(self):
        e = Event(1, 2)
        self.mode.press(e)
        assert self.mode._roi_tool.start_selection.call_count == 1

    def test_roi_updates_on_subsequent_presses(self):
        e = Event(1, 2)
        e2 = Event(1, 30)
        self.mode.press(e)
        self.mode._roi_tool.active.return_value = True
        self.mode.press(e2)
        assert self.mode._roi_tool.start_selection.call_count == 1
        assert self.mode._roi_tool.update_selection.call_count == 1

    def test_roi_finalizes_on_enter(self):
        e = Event(1, 2)
        e2 = Event(1, 20)
        e3 = Event(1, 30, key='enter')
        self.mode.press(e)
        self.mode._roi_tool.active.return_value = True
        self.mode.press(e2)
        self.mode.key(e3)
        self.mode._roi_tool.start_selection.assert_called_once_with(e)
        self.mode._roi_tool.update_selection.assert_called_once_with(e2)
        self.mode._roi_tool.finalize_selection.assert_called_once_with(e2)

    def test_roi_resets_on_escape(self):
        e = Event(1, 2)
        e2 = Event(1, 30, key='escape')
        self.mode.press(e)
        self.mode.key(e2)
        self.mode.press(e)
        assert self.mode._roi_tool.abort_selection.call_count == 1
        assert self.mode._roi_tool.start_selection.call_count == 2


class TestRectangleMode(TestRoiMode):

    def mode_factory(self):
        return RectangleMode


class TestCircleMode(TestRoiMode):

    def mode_factory(self):
        return CircleMode


class TestPolyMode(TestClickRoiMode):

    def mode_factory(self):
        return PolyMode


del TestRoiMode  # prevents test discovery from running abstract test
del TestClickRoiMode
