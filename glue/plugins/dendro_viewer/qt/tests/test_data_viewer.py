# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import os

import pytest
import numpy as np

from numpy.testing import assert_equal

from glue.core import Data
from glue.core.roi import PointROI
from glue.viewers.matplotlib.qt.tests.test_data_viewer import BaseTestMatplotlibDataViewer
from glue.app.qt import GlueApplication
from glue.core.state import GlueUnSerializer

from ..data_viewer import DendrogramViewer

DATA = os.path.join(os.path.dirname(__file__), 'data')


class TestDendrogramCommon(BaseTestMatplotlibDataViewer):

    viewer_cls = DendrogramViewer

    def init_data(self):
        return Data(label='d1', parent=[-1, 0, 1, 1], height=[1.3, 2.2, 3.2, 4.4])

    # TODO: Find a way to simplify having to overload this just because
    # we need to use a different ROI
    def test_apply_roi_undo(self):
        pass

        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)

        roi = PointROI(0.2, 2.8)
        self.viewer.apply_roi(roi)

        assert len(self.data.subsets) == 1

        mask1 = self.data.subsets[0].subset_state.to_mask(self.data)

        roi = PointROI(0.7, 2.8)
        self.viewer.apply_roi(roi)

        assert len(self.data.subsets) == 1

        mask2 = self.data.subsets[0].subset_state.to_mask(self.data)

        assert np.any(mask1 != mask2)

        self.application.undo()

        assert len(self.data.subsets) == 1

        mask3 = self.data.subsets[0].subset_state.to_mask(self.data)

        assert np.all(mask3 == mask1)

        self.application.redo()

        assert len(self.data.subsets) == 1

        mask4 = self.data.subsets[0].subset_state.to_mask(self.data)

        assert np.all(mask4 == mask2)

    @pytest.mark.skip
    def test_add_invalid_data():
        pass


class TestDendrogramViewer():

    def setup_method(self, method):

        self.data = Data(label='d1', parent=[-1, 0, 1, 1], height=[1.3, 2.2, 3.2, 4.4])

        self.app = GlueApplication()
        self.session = self.app.session
        self.hub = self.session.hub

        self.data_collection = self.session.data_collection
        self.data_collection.append(self.data)

        self.viewer = self.app.new_data_viewer(DendrogramViewer)

        self.data_collection.register_to_hub(self.hub)
        self.viewer.register_to_hub(self.hub)

    def teardown_method(self, method):
        self.viewer.close()
        self.viewer = None
        self.app.close()
        self.app = None

    def test_point_select(self):

        self.viewer.add_data(self.data)

        # By default selecting a structure selects all substructures
        roi = PointROI(0.5, 1.5)
        self.viewer.apply_roi(roi)
        assert len(self.data.subsets) == 1
        mask1 = self.data.subsets[0].subset_state.to_mask(self.data)
        assert_equal(mask1, [0, 1, 1, 1])

        # But this option can be turned off
        self.viewer.state.select_substruct = False
        self.viewer.apply_roi(roi)
        assert len(self.data.subsets) == 1
        mask1 = self.data.subsets[0].subset_state.to_mask(self.data)
        assert_equal(mask1, [0, 1, 0, 0])
        self.viewer.state.select_substruct = True

        # Try selecting a leaf
        roi = PointROI(0.2, 2.8)
        self.viewer.apply_roi(roi)
        assert len(self.data.subsets) == 1
        mask1 = self.data.subsets[0].subset_state.to_mask(self.data)
        assert_equal(mask1, [0, 0, 1, 0])

        # Try selecting another leaf
        roi = PointROI(0.7, 2.8)
        self.viewer.apply_roi(roi)
        assert len(self.data.subsets) == 1
        mask1 = self.data.subsets[0].subset_state.to_mask(self.data)
        assert_equal(mask1, [0, 0, 0, 1])

    def test_attribute_change_triggers_relayout(self):

        self.data.add_component([4, 5, 6, 7], 'flux')
        self.viewer.add_data(self.data)

        l = self.viewer.state._layout
        self.viewer.state.height_att = self.data.id['flux']
        assert self.viewer.state._layout is not l


class TestSessions(object):

    @pytest.mark.parametrize('protocol', [0, 1])
    def test_session_back_compat(self, protocol):

        filename = os.path.join(DATA, 'dendro_v{0}.glu'.format(protocol))

        with open(filename, 'r') as f:
            session = f.read()

        state = GlueUnSerializer.loads(session)

        ga = state.object('__main__')

        dc = ga.session.data_collection

        assert len(dc) == 1

        assert dc[0].label == 'data'

        viewer1 = ga.viewers[0][0]

        assert len(viewer1.state.layers) == 2

        assert viewer1.state.parent_att is dc[0].id['parent']
        assert viewer1.state.height_att is dc[0].id['height']
        assert viewer1.state.order_att is dc[0].id['height']

        layer_state = viewer1.state.layers[0]
        assert layer_state.visible
        assert layer_state.layer is dc[0]

        layer_state = viewer1.state.layers[1]
        assert layer_state.visible
        assert layer_state.layer is dc[0].subsets[0]

        ga.close()
