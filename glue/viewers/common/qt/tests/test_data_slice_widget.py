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
        s._ui_slider.button_prev.click()
        assert s.slice_center == 4
        s._ui_slider.button_next.click()
        s._ui_slider.button_next.click()
        assert s.slice_center == 6
        s._ui_slider.button_first.click()
        assert s.slice_center == 0
        s._ui_slider.button_prev.click()
        assert s.slice_center == 10
        s._ui_slider.button_next.click()
        assert s.slice_center == 0
        s._ui_slider.button_last.click()
        assert s.slice_center == 10
        s._ui_slider.button_next.click()
        assert s.slice_center == 0
        s._ui_slider.button_prev.click()
        assert s.slice_center == 10
        s._ui_slider.button_prev.click()
        assert s.slice_center == 9

    def test_slice_world(self):

        s = SliceWidget(lo=0, hi=5, world=[1, 3, 5, 5.5, 8, 12])

        # Check switching between world and pixel coordinates
        s.slice_center = 0
        assert s.slider_label == '1.0'
        s.use_world = False
        assert s.slider_label == '0'
        s.slice_center = 3
        assert s.slider_label == '3'
        s.use_world = True
        assert s.slider_label == '5.5'

        # Round to nearest
        s.slider_label = '11'
        assert s.slice_center == 5
        assert s.slider_label == '12.0'

        # Make sure out of bound values work
        s.slider_label = '20'
        assert s.slice_center == 5
        assert s.slider_label == '12.0'
        s.slider_label = '-10'
        assert s.slice_center == 0
        assert s.slider_label == '1.0'

        # And disable world and try and set by pixel
        s.use_world = False
        s.slider_label = '4'
        assert s.slice_center == 4
        assert s.slider_label == '4'


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
