import numpy as np
from unittest.mock import MagicMock
from numpy.testing import assert_allclose

from glue.core import Data
from glue.core.coordinates import IdentityCoordinates
from glue.app.qt import GlueApplication
from glue.tests.helpers import requires_astropy, requires_scipy
from glue.viewers.image.qt.data_viewer import ImageViewer

from ..pv_slicer import _slice_from_path, _slice_label, _slice_index


class TestPVSlicerMode(object):

    def setup_method(self, method):
        self.app = GlueApplication()
        self.data_collection = self.app.data_collection
        self.data = Data(x=np.arange(6000).reshape((10, 20, 30)))
        self.data_collection.append(self.data)
        self.viewer = self.app.new_data_viewer(ImageViewer, data=self.data)

    def teardown_method(self, method):
        self.viewer = None
        self.app.close()
        self.app = None

    def test_plain(self):

        self.viewer.toolbar.active_tool = 'slice'
        tool = self.viewer.toolbar.active_tool  # should now be set to PVSlicerMode

        roi = MagicMock()
        roi.to_polygon.return_value = [1, 10, 12], [2, 13, 14]

        mode = MagicMock()
        mode.roi.return_value = roi

        tool._extract_callback(mode)

        assert len(self.data_collection) == 2
        assert self.data_collection[1].shape == (10, 16)

        # If we do it again it should just update the existing slice dataset in-place

        roi.to_polygon.return_value = [1, 10, 12], [2, 13, 14]

        tool._extract_callback(mode)


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
    d = Data(x=np.zeros((2, 3, 4)), coords=IdentityCoordinates(n_dim=3))
    assert _slice_label(d, (0, 'y', 'x')) == 'World 0'
    assert _slice_label(d, ('y', 0, 'x')) == 'World 1'
    assert _slice_label(d, ('y', 'x', 0)) == 'World 2'


def test_slice_label_nocoords():
    d = Data(x=np.zeros((2, 3, 4)))
    assert _slice_label(d, (0, 'y', 'x')) == 'Pixel Axis 0 [z]'
    assert _slice_label(d, ('y', 0, 'x')) == 'Pixel Axis 1 [y]'
    assert _slice_label(d, ('y', 'x', 0)) == 'Pixel Axis 2 [x]'


def test_slice_index():
    d = Data(x=np.zeros((2, 3, 4)))
    assert _slice_index(d, (0, 'y', 'x')) == 0
    assert _slice_index(d, ('y', 0, 'x')) == 1
