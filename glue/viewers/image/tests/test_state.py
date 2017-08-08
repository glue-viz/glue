import numpy as np
from numpy.testing import assert_equal

from glue.core import Data

from ..state import ImageViewerState, ImageLayerState, AggregateSlice


class TestImageViewerState(object):

    def setup_method(self, method):
        self.state = ImageViewerState()

    def test_pixel_world_linking(self):

        data = Data(label='data', x=[[1, 2], [3, 4]], y=[[5, 6], [7, 8]])
        layer_state = ImageLayerState(layer=data, viewer_state=self.state)
        self.state.layers.append(layer_state)

        w1, w2 = data.world_component_ids
        p1, p2 = data.pixel_component_ids

        self.state.reference_data = data

        # Setting world components should set the pixel ones

        self.state.x_att_world = w1
        self.state.y_att_world = w2

        assert self.state.x_att is p1
        assert self.state.y_att is p2

        # Setting one component to the same as the other should trigger the other
        # to flip to prevent them from both being the same

        self.state.x_att_world = w2
        assert self.state.x_att is p2
        assert self.state.y_att is p1
        assert self.state.y_att_world is w1

        self.state.y_att_world = w2
        assert self.state.x_att is p1
        assert self.state.x_att_world is w1
        assert self.state.y_att is p2

        # Changing x_att and y_att should change the world equivalents

        self.state.x_att = p2
        assert self.state.x_att_world is w2
        assert self.state.y_att is p1
        assert self.state.y_att_world is w1

        self.state.y_att = p2
        assert self.state.y_att_world is w2
        assert self.state.x_att is p1
        assert self.state.x_att_world is w1

class TestSlicingAggregation():

    def setup_method(self, method):
        self.viewer_state = ImageViewerState()
        self.data = Data(x=np.ones((3, 4, 5, 6, 7)))
        self.layer_state = ImageLayerState(layer=self.data, viewer_state=self.viewer_state)
        self.viewer_state.layers.append(self.layer_state)
        self.p = self.data.pixel_component_ids

    def test_default(self):
        # Check default settings
        assert self.viewer_state.x_att == self.p[4]
        assert self.viewer_state.y_att == self.p[3]
        assert self.viewer_state.slices == (0, 0, 0, 0, 0)
        assert self.layer_state.get_sliced_data().shape == (6, 7)

    def test_flipped(self):
        # Make sure slice is transposed if needed
        self.viewer_state.x_att = self.p[3]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (0, 0, 0, 0, 0)
        assert self.layer_state.get_sliced_data().shape == (7, 6)

    def test_slice_preserved(self):
        # Make sure slice stays the same if changing attributes
        self.viewer_state.slices = (1, 3, 2, 5, 4)
        self.viewer_state.x_att = self.p[2]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (1, 3, 2, 5, 4)
        assert self.viewer_state.wcsaxes_slice == ['y', 5, 'x', 3, 1]
        assert self.layer_state.get_sliced_data().shape == (7, 5)
        self.viewer_state.x_att = self.p[2]
        self.viewer_state.y_att = self.p[1]
        assert self.viewer_state.slices == (1, 3, 2, 5, 4)
        assert self.viewer_state.wcsaxes_slice == [4, 5, 'x', 'y', 1]
        assert self.layer_state.get_sliced_data().shape == (4, 5)
        self.viewer_state.x_att = self.p[0]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (1, 3, 2, 5, 4)
        assert self.viewer_state.wcsaxes_slice == ['y', 5, 2, 3, 'x']
        assert self.layer_state.get_sliced_data().shape == (7, 3)

    def test_aggregation(self):
        # Check whether using AggregateSlice works
        slc1 = AggregateSlice(slice(None), 0, np.mean)
        slc2 = AggregateSlice(slice(2, 5), 3, np.sum)
        self.viewer_state.slices = (slc1, 3, 2, slc2, 4)
        self.viewer_state.x_att = self.p[2]
        self.viewer_state.y_att = self.p[4]
        assert self.viewer_state.slices == (slc1, 3, 2, slc2, 4)
        assert self.viewer_state.wcsaxes_slice == ['y', 3, 'x', 3, 0]
        result = self.layer_state.get_sliced_data()
        assert result.shape == (7, 5)
        assert_equal(result, 3)  # sum along 3 indices in one of the dimensions
