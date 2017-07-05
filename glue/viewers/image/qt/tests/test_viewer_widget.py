# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest

from astropy.wcs import WCS

import numpy as np
from numpy.testing import assert_allclose

from glue.core import Data
from glue.core.coordinates import Coordinates, WCSCoordinates
from glue.core.roi import RectangularROI
from glue.core.subset import RoiSubsetState, AndState
from glue import core
from glue.core.component_id import ComponentID
from glue.core.tests.util import simple_session
from glue.utils.qt import combo_as_string
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.viewers.image.state import ImageLayerState, ImageSubsetLayerState
from glue.viewers.scatter.state import ScatterLayerState

from ..data_viewer import ImageViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestImageCommon(BaseTestMatplotlibDataViewer):

    def init_data(self):
        return Data(label='d1', x=np.arange(12).reshape((3, 4)), y=np.ones((3, 4)))

    viewer_cls = ImageViewer

    @pytest.mark.skip()
    def test_double_add_ignored(self):
        pass


class MyCoords(Coordinates):
    def axis_label(self, i):
        return ['Banana', 'Apple'][i]


class TestImageViewer(object):

    def setup_method(self, method):

        self.coords = MyCoords()
        self.image1 = Data(label='image1', x=[[1, 2], [3, 4]], y=[[4, 5], [2, 3]])
        self.image2 = Data(label='image2', a=[[3, 3], [2, 2]], b=[[4, 4], [3, 2]], coords=self.coords)
        self.catalog = Data(label='catalog', c=[1, 3, 2], d=[4, 3, 3])
        self.hypercube = Data(label='hypercube', x=np.arange(120).reshape((2, 3, 4, 5)))

        self.session = simple_session()
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.image1)
        self.data_collection.append(self.image2)
        self.data_collection.append(self.catalog)
        self.data_collection.append(self.hypercube)

        self.viewer = ImageViewer(self.session)

        self.data_collection.register_to_hub(self.hub)
        self.viewer.register_to_hub(self.hub)

        self.options_widget = self.viewer.options_widget()


    def teardown_method(self, method):
        self.viewer.close()

    def test_basic(self):

        # Check defaults when we add data

        self.viewer.add_data(self.image1)

        assert combo_as_string(self.options_widget.ui.combodata_x_att_world) == 'World 0:World 1'
        assert combo_as_string(self.options_widget.ui.combodata_x_att_world) == 'World 0:World 1'

        assert self.viewer.axes.get_xlabel() == 'World 1'
        assert self.viewer.state.x_att_world is self.image1.id['World 1']
        assert self.viewer.state.x_att is self.image1.pixel_component_ids[1]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.x_min == -0.5
        # assert self.viewer.state.x_max == +1.5

        assert self.viewer.axes.get_ylabel() == 'World 0'
        assert self.viewer.state.y_att_world is self.image1.id['World 0']
        assert self.viewer.state.y_att is self.image1.pixel_component_ids[0]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.y_min == -0.5
        # assert self.viewer.state.y_max == +1.5

        assert not self.viewer.state.x_log
        assert not self.viewer.state.y_log

        assert len(self.viewer.state.layers) == 1

    def test_custom_coords(self):

        # Check defaults when we add data with coordinates

        self.viewer.add_data(self.image2)

        assert combo_as_string(self.options_widget.ui.combodata_x_att_world) == 'Banana:Apple'
        assert combo_as_string(self.options_widget.ui.combodata_x_att_world) == 'Banana:Apple'

        assert self.viewer.axes.get_xlabel() == 'Apple'
        assert self.viewer.state.x_att_world is self.image2.id['Apple']
        assert self.viewer.state.x_att is self.image2.pixel_component_ids[1]
        assert self.viewer.axes.get_ylabel() == 'Banana'
        assert self.viewer.state.y_att_world is self.image2.id['Banana']
        assert self.viewer.state.y_att is self.image2.pixel_component_ids[0]

    def test_flip(self):

        self.viewer.add_data(self.image1)

        x_min_start = self.viewer.state.x_min
        x_max_start = self.viewer.state.x_max

        self.options_widget.button_flip_x.click()

        assert self.viewer.state.x_min == x_max_start
        assert self.viewer.state.x_max == x_min_start

        y_min_start = self.viewer.state.y_min
        y_max_start = self.viewer.state.y_max

        self.options_widget.button_flip_y.click()

        assert self.viewer.state.y_min == y_max_start
        assert self.viewer.state.y_max == y_min_start

    def test_combo_updates_with_component_add(self):
        self.viewer.add_data(self.image1)
        self.image1.add_component([[9, 9], [8, 8]], 'z')
        assert self.viewer.state.x_att_world is self.image1.id['World 1']
        assert self.viewer.state.y_att_world is self.image1.id['World 0']
        # TODO: there should be an easier way to do this
        layer_style_editor = self.viewer._view.layout_style_widgets[self.viewer.layers[0]]
        assert combo_as_string(layer_style_editor.ui.combodata_attribute) == 'x:y:z'

    def test_apply_roi(self):

        self.viewer.add_data(self.image1)

        roi = RectangularROI(0.4, 1.6, -0.6, 0.6)

        assert len(self.viewer.layers) == 1

        self.viewer.apply_roi(roi)

        assert len(self.viewer.layers) == 2
        assert len(self.image1.subsets) == 1

        assert_allclose(self.image1.subsets[0].to_mask(), [[0, 1], [0, 0]])

        state = self.image1.subsets[0].subset_state
        assert isinstance(state, RoiSubsetState)

    def test_identical(self):

        # Check what happens if we set both attributes to the same coordinates

        self.viewer.add_data(self.image2)

        assert self.viewer.state.x_att_world is self.image2.id['Apple']
        assert self.viewer.state.y_att_world is self.image2.id['Banana']

        self.viewer.state.y_att_world = self.image2.id['Apple']

        assert self.viewer.state.x_att_world is self.image2.id['Banana']
        assert self.viewer.state.y_att_world is self.image2.id['Apple']

        self.viewer.state.x_att_world = self.image2.id['Apple']

        assert self.viewer.state.x_att_world is self.image2.id['Apple']
        assert self.viewer.state.y_att_world is self.image2.id['Banana']

    def test_aspect_subset(self):

        self.viewer.add_data(self.image1)

        assert self.viewer.state.aspect == 'equal'
        assert self.viewer.axes.get_aspect() == 'equal'

        self.viewer.state.aspect = 'auto'

        self.data_collection.new_subset_group('s1', self.image1.id['x'] > 0.)

        assert len(self.viewer.state.layers) == 2

        assert self.viewer.state.aspect == 'auto'
        assert self.viewer.axes.get_aspect() == 'auto'

        self.viewer.state.aspect = 'equal'

        self.data_collection.new_subset_group('s2', self.image1.id['x'] > 1.)

        assert len(self.viewer.state.layers) == 3

        assert self.viewer.state.aspect == 'equal'
        assert self.viewer.axes.get_aspect() == 'equal'

    def test_hypercube(self):

        # Check defaults when we add data

        self.viewer.add_data(self.hypercube)

        assert combo_as_string(self.options_widget.ui.combodata_x_att_world) == 'World 0:World 1:World 2:World 3'
        assert combo_as_string(self.options_widget.ui.combodata_x_att_world) == 'World 0:World 1:World 2:World 3'

        assert self.viewer.axes.get_xlabel() == 'World 3'
        assert self.viewer.state.x_att_world is self.hypercube.id['World 3']
        assert self.viewer.state.x_att is self.hypercube.pixel_component_ids[3]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.x_min == -0.5
        # assert self.viewer.state.x_max == +1.5

        assert self.viewer.axes.get_ylabel() == 'World 2'
        assert self.viewer.state.y_att_world is self.hypercube.id['World 2']
        assert self.viewer.state.y_att is self.hypercube.pixel_component_ids[2]
        # TODO: make sure limits are deterministic then update this
        # assert self.viewer.state.y_min == -0.5
        # assert self.viewer.state.y_max == +1.5

        assert not self.viewer.state.x_log
        assert not self.viewer.state.y_log

        assert len(self.viewer.state.layers) == 1

    def test_hypercube_world(self):

        # Check defaults when we add data

        wcs = WCS(naxis=4)
        hypercube2 = Data()
        hypercube2.coords = WCSCoordinates(wcs=wcs)
        hypercube2.add_component(np.random.random((2, 3, 4, 5)), 'a')

        self.data_collection.append(hypercube2)

        self.viewer.add_data(hypercube2)


class TestSessions(object):

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_back_compat(self, protocol):

        filename = os.path.join(DATA, 'image_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 2

        assert dc[0].label == 'data1'
        assert dc[1].label == 'data2'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 3

        assert viewer1.state.x_att_world is dc[0].id['World 1']
        assert viewer1.state.y_att_world is dc[0].id['World 0']

        if protocol == 0:
            assert viewer1.state.x_min < 0.
            assert viewer1.state.x_max > 1.5
            assert_allclose(viewer1.state.y_min, 0)
            assert_allclose(viewer1.state.y_max, 2)
        else:
            assert_allclose(viewer1.state.x_min, -0.936)
            assert_allclose(viewer1.state.x_max, +1.937)
            assert_allclose(viewer1.state.y_min, -0.6121290)
            assert_allclose(viewer1.state.y_max, +1.6121290)

        layer_state = viewer1.state.layers[0]
        assert isinstance(layer_state, ImageLayerState)
        assert layer_state.visible
        assert layer_state.bias == 0.5
        assert layer_state.contrast == 1.0
        assert layer_state.stretch == 'sqrt'
        assert layer_state.percentile == 99

        layer_state = viewer1.state.layers[1]
        assert isinstance(layer_state, ScatterLayerState)
        assert layer_state.visible

        layer_state = viewer1.state.layers[2]
        assert isinstance(layer_state, ImageSubsetLayerState)
        assert not layer_state.visible

        viewer2 = ga.viewers[0][1]

        assert len(viewer2.state.layers) == 2

        assert viewer2.state.x_att_world is dc[0].id['World 1']
        assert viewer2.state.y_att_world is dc[0].id['World 0']

        if protocol == 0:
            assert viewer2.state.x_min < 0.
            assert viewer2.state.x_max > 1.5
            assert_allclose(viewer2.state.y_min, 0)
            assert_allclose(viewer2.state.y_max, 2)
        else:
            assert_allclose(viewer1.state.x_min, -0.936)
            assert_allclose(viewer1.state.x_max, +1.937)
            assert_allclose(viewer1.state.y_min, -0.6121290)
            assert_allclose(viewer1.state.y_max, +1.6121290)


        layer_state = viewer2.state.layers[0]
        assert layer_state.visible
        assert layer_state.stretch == 'arcsinh'
        assert layer_state.v_min == 1
        assert layer_state.v_max == 4

        layer_state = viewer2.state.layers[1]
        assert layer_state.visible

        viewer3 = ga.viewers[0][2]

        assert len(viewer3.state.layers) == 2

        assert viewer3.state.x_att_world is dc[0].id['World 1']
        assert viewer3.state.y_att_world is dc[0].id['World 0']

        if protocol == 0:
            assert viewer3.state.x_min < 0.0
            assert viewer3.state.x_max > 1.5
            assert_allclose(viewer3.state.y_min, 0)
            assert_allclose(viewer3.state.y_max, 2)
        else:
            assert_allclose(viewer1.state.x_min, -0.936)
            assert_allclose(viewer1.state.x_max, +1.937)
            assert_allclose(viewer1.state.y_min, -0.6121290)
            assert_allclose(viewer1.state.y_max, +1.6121290)

        layer_state = viewer3.state.layers[0]
        assert layer_state.visible
        assert layer_state.stretch == 'linear'
        assert layer_state.v_min == -2
        assert layer_state.v_max == 2

        layer_state = viewer3.state.layers[1]
        assert layer_state.visible

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_cube_back_compat(self, protocol):

        filename = os.path.join(DATA, 'image_cube_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'array'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 1

        assert viewer1.state.x_att_world is dc[0].id['World 2']
        assert viewer1.state.y_att_world is dc[0].id['World 1']
        assert viewer1.state.slices == [2, 0, 0, 1]

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_rgb_back_compat(self, protocol):

        filename = os.path.join(DATA, 'image_rgb_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'rgbcube'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 3
        assert viewer1.state.color_mode == 'One color per layer'

        layer_state = viewer1.state.layers[0]
        assert layer_state.visible
        assert layer_state.attribute.label == 'a'
        assert layer_state.color == 'r'

        layer_state = viewer1.state.layers[1]
        assert not layer_state.visible
        assert layer_state.attribute.label == 'c'
        assert layer_state.color == 'g'

        layer_state = viewer1.state.layers[2]
        assert layer_state.visible
        assert layer_state.attribute.label == 'b'
        assert layer_state.color == 'b'
