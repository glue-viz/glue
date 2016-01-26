from __future__ import absolute_import, division, print_function

from collections import namedtuple

import pytest
import numpy as np
from mock import MagicMock

from ..profile_viewer import ProfileViewer
from glue.utils import renderless_figure


FIG = renderless_figure()
Event = namedtuple('Event', 'xdata ydata inaxes button dblclick')


class TestProfileViewer(object):

    def setup_method(self, method):
        FIG.clf()
        FIG.canvas.draw = MagicMock()
        self.viewer = ProfileViewer(FIG)
        self.axes = self.viewer.axes

    def test_set_profile(self):
        self.viewer.set_profile([1, 2, 3], [2, 3, 4])
        self.axes.figure.canvas.draw.assert_called_once_with()

    def test_new_value_callback_fire(self):
        cb = MagicMock()
        s = self.viewer.new_value_grip(callback=cb)
        s.value = 20
        cb.assert_called_once_with(20)

    def test_new_range_callback_fire(self):
        cb = MagicMock()
        s = self.viewer.new_range_grip(callback=cb)
        s.range = (20, 40)
        cb.assert_called_once_with((20, 40))

    def test_pick_grip(self):
        self.viewer.set_profile([1, 2, 3], [10, 20, 30])
        s = self.viewer.new_value_grip()

        s.value = 1.7
        assert self.viewer.pick_grip(1.7, 20) is s

    def test_pick_grip_false(self):
        self.viewer.set_profile([1, 2, 3], [10, 20, 30])
        s = self.viewer.new_value_grip()

        s.value = 3
        assert self.viewer.pick_grip(1.7, 20) is None

    def test_pick_range_grip(self):
        self.viewer.set_profile([1, 2, 3], [10, 20, 30])
        s = self.viewer.new_range_grip()

        s.range = (1.5, 2.5)

        assert self.viewer.pick_grip(1.5, 20) is s
        assert self.viewer.pick_grip(2.5, 20) is s
        assert self.viewer.pick_grip(1.0, 20) is None

    def test_value_drag_updates_value(self):
        h = self.viewer.new_value_grip()
        x2 = h.value + 10

        self._click(h.value)
        self._drag(x2)
        self._release()

        assert h.value == x2

    def test_disabled_grips_ignore_events(self):
        h = self.viewer.new_value_grip()
        h.value = 5

        h.disable()

        self._click(h.value)
        self._drag(10)
        self._release()

        assert h.value == 5

    def test_value_ignores_distant_picks(self):
        self.viewer.set_profile([1, 2, 3], [1, 2, 3])
        h = self.viewer.new_value_grip()
        h.value = 3

        self._click(1)
        self._drag(2)
        self._release()

        assert h.value == 3

    def test_range_translates_on_center_drag(self):
        h = self.viewer.new_range_grip()
        h.range = (1, 3)
        self._click_range_center(h)
        self._drag(1)
        self._release()
        assert h.range == (0, 2)

    def test_range_stretches_on_edge_drag(self):
        h = self.viewer.new_range_grip()
        h.range = (1, 3)

        self._click(1)
        self._drag(2)
        self._release()
        assert h.range == (2, 3)

    def test_range_redefines_on_distant_drag(self):
        self.viewer.set_profile([1, 2, 3], [1, 2, 3])
        h = self.viewer.new_range_grip()
        h.range = (2, 2)

        self._click(1)
        self._drag(1.5)
        self._release()

        assert h.range == (1, 1.5)

    def test_dblclick_sets_value(self):
        h = self.viewer.new_value_grip()
        h.value = 1

        self._click(1.5, double=True)
        assert h.value == 1.5

    def _click_range_center(self, grip):
        x, y = sum(grip.range) / 2, 0
        self._click(x, y)

    def _click(self, x, y=0, double=False):
        e = Event(xdata=x, ydata=y, inaxes=True, button=1, dblclick=double)
        self.viewer._on_down(e)

    def _drag(self, x, y=0):
        e = Event(xdata=x, ydata=y, inaxes=True, button=1, dblclick=False)
        self.viewer._on_move(e)

    def _release(self):
        e = Event(xdata=0, ydata=0, inaxes=True, button=1, dblclick=False)
        self.viewer._on_up(e)

    def test_fit(self):
        fitter = MagicMock()
        self.viewer.set_profile([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        self.viewer.fit(fitter, xlim=[1, 3])

        args = fitter.build_and_fit.call_args[0]
        np.testing.assert_array_equal(args[0], [1, 2, 3])
        np.testing.assert_array_equal(args[1], [2, 3, 4])

    def test_fit_error_without_profile(self):
        with pytest.raises(ValueError) as exc:
            self.viewer.fit(None)
        assert exc.value.args[0] == "Must set profile before fitting"

    def test_new_select(self):
        h = self.viewer.new_range_grip()

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
