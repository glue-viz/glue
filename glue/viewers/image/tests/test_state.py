from glue.core import Data

from ..state import ImageViewerState

class TestImageViewerState(object):

    def setup_method(self, method):
        self.state = ImageViewerState()
        self.data = Data(label='data', x=[[1, 2], [3, 4]], y=[[5, 6], [7, 8]])

    def test_pixel_world_linking(self):

        w1, w2 = self.data.world_component_ids
        p1, p2 = self.data.pixel_component_ids

        self.state.reference_data = self.data

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
