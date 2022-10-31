import numpy as np
from unittest.mock import MagicMock
from numpy.testing import assert_allclose

from glue.core import Data
from glue.core.coordinates import IdentityCoordinates
from glue.viewers.image.qt import StandaloneImageViewer, ImageViewer
from glue.tests.helpers import requires_astropy, requires_scipy
from glue.app.qt import GlueApplication
from glue.utils.qt import process_events

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


class TestStandaloneImageViewer(object):

    def setup_method(self, method):
        im = np.random.random((3, 3))
        self.w = StandaloneImageViewer(im)

    def teardown_method(self, method):
        self.w.close()

    def test_set_cmap(self):
        cm_mode = self.w.toolbar.tools['image:colormap']
        act = cm_mode.menu_actions()[1]
        act.trigger()
        assert self.w._composite.layers['image']['cmap'] is act.cmap

    def test_double_set_image(self):
        assert len(self.w._axes.images) == 1
        self.w.set_image(np.zeros((3, 3)))
        assert len(self.w._axes.images) == 1


class MockImageViewer(object):

    def __init__(self, slice, data):
        self.slice = slice
        self.data = data
        self.wcs = None
        self.state = MagicMock()


class TestPVSliceWidget(object):

    def setup_method(self, method):

        self.d = Data(x=np.zeros((2, 3, 4)))
        self.slc = (0, 'y', 'x')
        self.image = MockImageViewer(self.slc, self.d)
        self.w = PVSliceWidget(image=np.zeros((3, 4)), wcs=None, image_viewer=self.image)

    def teardown_method(self, method):
        self.w.close()

    def test_basic(self):
        pass


class TestPVSliceTool(object):

    def setup_method(self, method):
        self.cube = Data(label='cube', x=np.arange(1000).reshape((5, 10, 20)))
        self.application = GlueApplication()
        self.application.data_collection.append(self.cube)
        self.viewer = self.application.new_data_viewer(ImageViewer)
        self.viewer.add_data(self.cube)

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.application.close()
        self.application = None

    @requires_astropy
    @requires_scipy
    def test_basic(self):

        self.viewer.toolbar.active_tool = 'slice'

        self.viewer.axes.figure.canvas.draw()
        process_events()

        x, y = self.viewer.axes.transData.transform([[0.9, 4]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)
        x, y = self.viewer.axes.transData.transform([[7.2, 6.6]])[0]
        self.viewer.axes.figure.canvas.button_press_event(x, y, 1)

        process_events()

        assert len(self.application.tab().subWindowList()) == 1

        self.viewer.axes.figure.canvas.key_press_event('enter')

        process_events()

        assert len(self.application.tab().subWindowList()) == 2

        pv_widget = self.application.tab().subWindowList()[1].widget()
        assert pv_widget._x.shape == (6,)
        assert pv_widget._y.shape == (6,)
