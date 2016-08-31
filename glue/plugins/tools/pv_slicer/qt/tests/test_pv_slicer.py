from __future__ import absolute_import, division, print_function

import numpy as np
from mock import MagicMock
from numpy.testing import assert_allclose

from glue.core import Data
from glue.viewers.image.qt import StandaloneImageWidget
from glue.tests.helpers import requires_astropy, requires_scipy

from ..pv_slicer import _slice_from_path, _slice_label, _slice_index, PVSliceWidget


@requires_astropy
@requires_scipy
class TestSliceExtraction(object):

    def setup_method(self, method):
        self.x = np.random.random((2, 3, 4))
        self.d = Data(x=self.x)

    def test_constant_y(self):

        slc = (0, 'y', 'x')
        x = [-0.5, 3.5]
        y = [0, 0]
        s = _slice_from_path(x, y, self.d, 'x', slc)[0]
        assert_allclose(s, self.x[:, 0, :])

    def test_constant_x(self):

        slc = (0, 'y', 'x')
        y = [-0.5, 2.5]
        x = [0, 0]
        s = _slice_from_path(x, y, self.d, 'x', slc)[0]
        assert_allclose(s, self.x[:, :, 0])

    def test_transpose(self):
        slc = (0, 'x', 'y')
        y = [-0.5, 3.5]
        x = [0, 0]
        s = _slice_from_path(x, y, self.d, 'x', slc)[0]
        assert_allclose(s, self.x[:, 0, :])


def test_slice_label():
    d = Data(x=np.zeros((2, 3, 4)))
    assert _slice_label(d, (0, 'y', 'x')) == 'World 0'
    assert _slice_label(d, ('y', 0, 'x')) == 'World 1'
    assert _slice_label(d, ('y', 'x', 0)) == 'World 2'


def test_slice_index():
    d = Data(x=np.zeros((2, 3, 4, 1)))
    assert _slice_index(d, (0, 'y', 'x', 0)) == 0
    assert _slice_index(d, (0, 'y', 0, 'x')) == 2


class TestStandaloneImageWidget(object):

    def setup_method(self, method):
        im = np.random.random((3, 3))
        self.w = StandaloneImageWidget(im)

    def test_set_cmap(self):
        cm_mode = self.w.toolbar.tools['image:colormap']
        act = cm_mode.menu_actions()[1]
        act.trigger()
        assert self.w._im.cmap is act.cmap

    def test_double_set_image(self):
        assert len(self.w._axes.images) == 1
        self.w.set_image(np.zeros((3, 3)))
        assert len(self.w._axes.images) == 1


class MockImageWidget(object):

    def __init__(self, slice, data):
        self.slice = slice
        self.data = data
        self.wcs = None
        self.client = MagicMock()


class TestPVSliceWidget(object):

    def setup_method(self, method):

        self.d = Data(x=np.zeros((2, 3, 4)))
        self.slc = (0, 'y', 'x')
        self.image = MockImageWidget(self.slc, self.d)
        self.w = PVSliceWidget(image=np.zeros((3, 4)), wcs=None, image_client=self.image.client)

    def test_basic(self):
        pass
