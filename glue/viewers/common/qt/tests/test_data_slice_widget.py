from __future__ import absolute_import, division, print_function

import numpy as np
from mock import MagicMock

from glue import core

from ..data_slice_widget import SliceWidget, DataSlice


class TestSliceWidget(object):

    def test_slice_center(self):
        s = SliceWidget(lo=0, hi=10)
        assert s.slice_center == 5

    def test_browse_slice(self):
        s = SliceWidget(lo=0, hi=10)
        assert s.slice_center == 5
        s._ui_slider.prev.click()
        assert s.slice_center == 4
        s._ui_slider.next.click()
        s._ui_slider.next.click()
        assert s.slice_center == 6
        s._ui_slider.first.click()
        assert s.slice_center == 0
        s._ui_slider.prev.click()
        assert s.slice_center == 0
        s._ui_slider.last.click()
        assert s.slice_center == 10
        s._ui_slider.next.click()
        assert s.slice_center == 10
        s._ui_slider.prev.click()
        assert s.slice_center == 9


class TestArraySlice(object):

    def test_1d(self):
        d = core.Data(x=[1, 2, 3])
        s = DataSlice(d)
        assert s.slice == ('x',)

    def test_2d(self):
        d = core.Data(x=[[1]])
        s = DataSlice(d)
        assert s.slice == ('y', 'x')

    def test_3d(self):
        d = core.Data(x=np.zeros((3, 3, 3)))
        s = DataSlice(d)

        assert s.slice == (1, 'y', 'x')

    def test_3d_change_mode(self):
        d = core.Data(x=np.zeros((3, 4, 5)))
        s = DataSlice(d)
        changed = MagicMock()
        s.slice_changed.connect(changed)

        assert s.slice == (1, 'y', 'x')

        s._slices[1].mode = 'slice'
        assert s.slice == ('y', 1, 'x')
        assert changed.call_count == 1

        s._slices[2].mode = 'slice'
        assert s.slice == ('y', 'x', 2)
        assert changed.call_count == 2

        s._slices[2].mode = 'y'
        assert s.slice == (1, 'x', 'y')
        assert changed.call_count == 3

        s._slices[2].mode = 'x'
        assert s.slice == (1, 'y', 'x')
        assert changed.call_count == 4

    def test_3d_change_slice(self):
        d = core.Data(x=np.zeros((3, 4, 5)))
        s = DataSlice(d)
        changed = MagicMock()
        s.slice_changed.connect(changed)

        s._slices[0].slice_center = 2
        assert s.slice == (2, 'y', 'x')
        assert changed.call_count == 1

        s._slices[1].mode = 'slice'
        s._slices[1].slice_center = 0
        assert s.slice == ('y', 0, 'x')
        assert changed.call_count == 3

        s._slices[2].mode = 'slice'
        assert s.slice == ('y', 'x', 2)
        assert changed.call_count == 4
