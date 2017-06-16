# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest

import numpy as np
from numpy.testing import assert_allclose

from glue.core import Data
from glue.core.roi import RectangularROI
from glue.core.subset import RoiSubsetState, AndState
from glue import core
from glue.core.component_id import ComponentID
from glue.core.tests.util import simple_session
from glue.utils.qt import combo_as_string
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.core.state import GlueUnSerializer
from glue.viewers.image_new.state import ImageLayerState, ImageSubsetLayerState
from glue.viewers.scatter.state import ScatterLayerState

from ..data_viewer import ImageViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestImageCommon(BaseTestMatplotlibDataViewer):
    def init_data(self):
        return Data(label='d1', x=np.arange(12).reshape((3, 4)), y=np.ones((3, 4)))
    viewer_cls = ImageViewer



class TestImageViewer(object):

    def setup_method(self, method):

        self.data1 = Data(label='d1', a=[[1, 2], [3, 4]], b=[[5, 6], [7, 8]])
        self.data2 = Data(label='d12', x=[[2, 1], [1, 1]], b=[[3, 3], [2, 2]])

        self.session = simple_session()
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data1)
        self.data_collection.append(self.data2)

        self.viewer = ImageViewer(self.session)

        self.data_collection.register_to_hub(self.hub)
        self.viewer.register_to_hub(self.hub)

    def teardown_method(self, method):
        self.viewer.close()

    @pytest.mark.parametrize('protocol', [0])
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

        delta = 0.3419913419913423

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 3

        assert viewer1.state.x_att_world is dc[0].id['World 1']
        assert viewer1.state.y_att_world is dc[0].id['World 0']

        assert_allclose(viewer1.state.x_min, -delta)
        assert_allclose(viewer1.state.x_max, 2 + delta)
        assert_allclose(viewer1.state.y_min, 0)
        assert_allclose(viewer1.state.y_max, 2)

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

        assert_allclose(viewer2.state.x_min, -delta)
        assert_allclose(viewer2.state.x_max, 2 + delta)
        assert_allclose(viewer2.state.y_min, 0)
        assert_allclose(viewer2.state.y_max, 2)

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

        assert_allclose(viewer3.state.x_min, -delta)
        assert_allclose(viewer3.state.x_max, 2 + delta)
        assert_allclose(viewer3.state.y_min, 0)
        assert_allclose(viewer3.state.y_max, 2)

        layer_state = viewer3.state.layers[0]
        assert layer_state.visible
        assert layer_state.stretch == 'linear'
        assert layer_state.v_min == -2
        assert layer_state.v_max == 2

        layer_state = viewer3.state.layers[1]
        assert layer_state.visible
