from collections import namedtuple

from mock import MagicMock

from ..profile_viewer import ProfileViewer
from .util import renderless_figure

FIG = renderless_figure()
Event = namedtuple('Event', 'xdata ydata inaxes')

class TestProfileViewer(object):

    def setup_method(self, method):
        FIG.clf()
        FIG.canvas.draw = MagicMock()
        self.axes = FIG.add_subplot(111)
        self.viewer = ProfileViewer(self.axes)


    def test_set_profile(self):
        artist = self.viewer.set_profile([1,2,3], [2,3,4])
        self.axes.figure.canvas.draw.assert_called_once_with()

    def test_new_slider_callback_fire(self):
        cb = MagicMock()
        s = self.viewer.new_slider_handle(callback=cb)
        s.value = 20
        cb.assert_called_once_with(20)

    def test_new_range_callback_fire(self):
        cb = MagicMock()
        s = self.viewer.new_range_handle(callback=cb)
        s.range = (20, 40)
        cb.assert_called_once_with((20, 40))

    def test_pick_handle(self):
        self.viewer.set_profile([1, 2, 3], [10, 20, 30])
        s = self.viewer.new_slider_handle()

        s.value = 1.7
        assert self.viewer.pick_handle(1.7, 20) is s

    def test_pick_handle_false(self):
        self.viewer.set_profile([1, 2, 3], [10, 20, 30])
        s = self.viewer.new_slider_handle()

        s.value = 3
        assert self.viewer.pick_handle(1.7, 20) is None

    def test_pick_range_handle(self):
        self.viewer.set_profile([1, 2, 3], [10, 20, 30])
        s = self.viewer.new_range_handle()

        s.range = (1.5, 2.5)

        assert self.viewer.pick_handle(1.5, 20) is s
        assert self.viewer.pick_handle(2.5, 20) is s
        assert self.viewer.pick_handle(1.0, 20) is None

    def test_slider_drag_updates_value(self):
        h = self.viewer.new_slider_handle()
        x2 = h.value + 10

        self._click(h.value)
        self._drag(x2)
        self._release()

        assert h.value == x2

    def test_range_translates_on_center_drag(self):
        h = self.viewer.new_range_handle()
        h.range = (1, 3)
        self._click_range_center(h)
        self._drag(1)
        self._release()
        assert h.range == (0, 2)

    def test_range_stretches_on_edge_drag(self):
        h = self.viewer.new_range_handle()
        h.range = (1, 3)

        self._click(1)
        self._drag(2)
        self._release()
        assert h.range == (2, 3)

    def _click_range_center(self, handle):
        x, y = sum(handle.range) / 2, 0
        self._click(x, y)

    def _click(self, x, y=0):
        e = Event(xdata=x, ydata=y, inaxes=True)
        self.viewer._on_down(e)

    def _drag(self, x, y=0):
        e = Event(xdata=x, ydata=y, inaxes=True)
        self.viewer._on_move(e)

    def _release(self):
        e = Event(xdata=0, ydata=0, inaxes=True)
        self.viewer._on_up(e)
