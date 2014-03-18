from collections import namedtuple

from mock import MagicMock
import pytest
import numpy as np

from ..profile_viewer import ProfileViewer
from .util import renderless_figure

FIG = renderless_figure()
Event = namedtuple('Event', 'xdata ydata inaxes button')


class TestProfileViewer(object):

    def setup_method(self, method):
        FIG.clf()
        FIG.canvas.draw = MagicMock()
        self.axes = FIG.add_subplot(111)
        self.viewer = ProfileViewer(self.axes)

    def test_set_profile(self):
        self.viewer.set_profile([1, 2, 3], [2, 3, 4])
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

    def test_disabled_handles_ignore_events(self):
        h = self.viewer.new_slider_handle()
        h.value = 5

        h.disable()

        self._click(h.value)
        self._drag(10)
        self._release()

        assert h.value == 5

    def test_slider_ignores_distant_picks(self):
        self.viewer.set_profile([1, 2, 3], [1, 2, 3])
        h = self.viewer.new_slider_handle()
        h.value = 3

        self._click(1)
        self._drag(2)
        self._release()

        assert h.value == 3

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

    def test_range_redefines_on_distant_drag(self):
        self.viewer.set_profile([1, 2, 3], [1, 2, 3])
        h = self.viewer.new_range_handle()
        h.range = (2, 2)

        self._click(1)
        self._drag(1.5)
        self._release()

        assert h.range == (1, 1.5)

    def _click_range_center(self, handle):
        x, y = sum(handle.range) / 2, 0
        self._click(x, y)

    def _click(self, x, y=0):
        e = Event(xdata=x, ydata=y, inaxes=True, button=1)
        self.viewer._on_down(e)

    def _drag(self, x, y=0):
        e = Event(xdata=x, ydata=y, inaxes=True, button=1)
        self.viewer._on_move(e)

    def _release(self):
        e = Event(xdata=0, ydata=0, inaxes=True, button=1)
        self.viewer._on_up(e)

    def test_fit(self):
        fitter = MagicMock()
        self.viewer.set_profile([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        self.viewer.fit(fitter, xlim=[1, 3], plot=False)

        args = fitter.fit.call_args[0]
        np.testing.assert_array_equal(args[0], [1, 2, 3])
        np.testing.assert_array_equal(args[1], [2, 3, 4])

    def test_fit_error_without_profile(self):
        with pytest.raises(ValueError) as exc:
            self.viewer.fit(None)
        assert exc.value.args[0] == "Must set profile before fitting"

    def test_fit_with_plotting(self):
        fitter = MagicMock()
        self.viewer._plot_fit = MagicMock()
        self.viewer.set_profile([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        self.viewer.fit(fitter, xlim=[1, 3], plot=True)
        self.viewer._plot_fit.assert_called_once_with(fitter)

    def test_new_select(self):
        h = self.viewer.new_range_handle()

        h.new_select(0, 1)
        h.new_drag(1, 1)
        h.release()
        assert h.range == (0, 1)

        h.new_select(1, 1)
        h.new_drag(.5, 1)
        h.release()
        assert h.range == (0.5, 1)

        h.new_select(.4, 1)
        h.new_drag(.4, 1)
        h.release()
        assert h.range == (.4, .4)
