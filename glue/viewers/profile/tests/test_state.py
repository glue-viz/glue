from __future__ import absolute_import, division, print_function

import numpy as np

from numpy.testing import assert_allclose

from glue.core import Data, Coordinates
from glue.core.tests.test_state import clone
from glue.utils import nanmean, nanmedian, nansum, nanmin, nanmax

from ..state import ProfileViewerState, ProfileLayerState


class SimpleCoordinates(Coordinates):

    def world2pixel(self, *world):
        return tuple([0.5 * w for w in world])

    def pixel2world(self, *pixel):
        return tuple([2 * p for p in pixel])


class TestProfileViewerState:

    def setup_method(self, method):
        self.data = Data(label='d1')
        self.data.coords = SimpleCoordinates()
        self.data['x'] = np.arange(24).reshape((3, 4, 2)).astype(float)
        self.viewer_state = ProfileViewerState()
        self.layer_state = ProfileLayerState(viewer_state=self.viewer_state,
                                             layer=self.data)
        self.viewer_state.layers.append(self.layer_state)
        self.viewer_state.function = nanmean

    def test_basic(self):
        x, y = self.layer_state.profile
        assert_allclose(x, [0, 2, 4])
        assert_allclose(y, [3.5, 11.5, 19.5])

    def test_basic_world(self):
        self.viewer_state.x_att = self.data.world_component_ids[0]
        x, y = self.layer_state.profile
        assert_allclose(x, [0, 2, 4])
        assert_allclose(y, [3.5, 11.5, 19.5])

    def test_x_att(self):

        self.viewer_state.x_att = self.data.pixel_component_ids[0]
        x, y = self.layer_state.profile
        assert_allclose(x, [0, 1, 2])
        assert_allclose(y, [3.5, 11.5, 19.5])

        self.viewer_state.x_att = self.data.pixel_component_ids[1]
        x, y = self.layer_state.profile
        assert_allclose(x, [0, 1, 2, 3])
        assert_allclose(y, [8.5, 10.5, 12.5, 14.5])

        self.viewer_state.x_att = self.data.pixel_component_ids[2]
        x, y = self.layer_state.profile
        assert_allclose(x, [0, 1])
        assert_allclose(y, [11, 12])

    def test_function(self):

        self.viewer_state.function = nanmean
        x, y = self.layer_state.profile
        assert_allclose(y, [3.5, 11.5, 19.5])

        self.viewer_state.function = nanmin
        x, y = self.layer_state.profile
        assert_allclose(y, [0, 8, 16])

        self.viewer_state.function = nanmax
        x, y = self.layer_state.profile
        assert_allclose(y, [7, 15, 23])

        self.viewer_state.function = nansum
        x, y = self.layer_state.profile
        assert_allclose(y, [28, 92, 156])

        self.viewer_state.function = nanmedian
        x, y = self.layer_state.profile
        assert_allclose(y, [3.5, 11.5, 19.5])

    def test_subset(self):

        subset = self.data.new_subset()
        subset.subset_state = self.data.id['x'] > 10

        self.layer_state.layer = subset

        x, y = self.layer_state.profile
        assert_allclose(x, [0, 2, 4])
        assert_allclose(y, [np.nan, 13., 19.5])

        subset.subset_state = self.data.id['x'] > 100

        # TODO: the fact we have to call this isn't ideal
        self.layer_state._update_profile()
        x, y = self.layer_state.profile
        assert len(x) == 0
        assert len(y) == 0

    def test_clone(self):

        self.viewer_state.x_att = self.data.pixel_component_ids[1]
        self.viewer_state.function = nanmedian

        self.layer_state.attribute = self.data.id['x']
        self.layer_state.linewidth = 3

        viewer_state_new = clone(self.viewer_state)

        assert viewer_state_new.x_att.label == 'Pixel Axis 1 [y]'
        assert viewer_state_new.function is nanmedian

        assert self.layer_state.attribute.label == 'x'
        assert self.layer_state.linewidth == 3

    def test_limits(self):

        self.viewer_state.x_att = self.data.pixel_component_ids[0]

        assert self.viewer_state.x_min == -0.5
        assert self.viewer_state.x_max == 2.5

        self.viewer_state.flip_x()

        assert self.viewer_state.x_min == 2.5
        assert self.viewer_state.x_max == -0.5

        self.viewer_state.x_min = 1
        self.viewer_state.x_max = 1.5

        assert self.viewer_state.x_min == 1
        assert self.viewer_state.x_max == 1.5

        self.viewer_state.reset_limits()

        assert self.viewer_state.x_min == -0.5
        assert self.viewer_state.x_max == 2.5
