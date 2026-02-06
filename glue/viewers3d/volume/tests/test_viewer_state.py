import numpy as np
from numpy.testing import assert_allclose

from glue.core import Data, DataCollection
from glue.core.tests.test_state import clone

from ..viewer_state import VolumeViewerState3D


class MockLayerState:
    """Simple mock layer state for testing viewer state."""
    def __init__(self, layer):
        self.layer = layer


class TestVolumeViewerState3D:

    def setup_method(self, method):
        self.data = Data(label='test_cube')
        self.data['x'] = np.arange(24).reshape((2, 3, 4))
        self.data_collection = DataCollection([self.data])
        self.state = VolumeViewerState3D()

    def test_adding_layer_sets_reference_data(self):
        layer_state = MockLayerState(self.data)
        self.state.layers.append(layer_state)
        assert self.state.reference_data == self.data

    def test_adding_layer_initializes_slices(self):
        layer_state = MockLayerState(self.data)
        self.state.layers.append(layer_state)
        assert len(self.state.slices) == self.data.ndim

    def test_x_att_swap_with_y(self):
        self.state.layers.append(MockLayerState(self.data))
        pixel_ids = self.data.pixel_component_ids
        self.state.x_att = pixel_ids[0]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[2]

        self.state.x_att = pixel_ids[1]
        assert self.state.y_att == pixel_ids[0]

    def test_x_att_swap_with_z(self):
        self.state.layers.append(MockLayerState(self.data))
        pixel_ids = self.data.pixel_component_ids
        self.state.x_att = pixel_ids[0]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[2]

        self.state.x_att = pixel_ids[2]
        assert self.state.z_att == pixel_ids[0]

    def test_y_att_swap_with_z(self):
        self.state.layers.append(MockLayerState(self.data))
        pixel_ids = self.data.pixel_component_ids
        self.state.x_att = pixel_ids[0]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[2]

        self.state.y_att = pixel_ids[2]
        assert self.state.z_att == pixel_ids[1]

    def test_clip_limits_relative_full_range(self):
        data = Data(label='cube')
        data['x'] = np.arange(120).reshape((3, 4, 10))
        self.state.layers.append(MockLayerState(data))
        pixel_ids = data.pixel_component_ids
        self.state.x_att = pixel_ids[2]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[0]

        self.state.set_limits(0, 10, 0, 4, 0, 3)
        assert_allclose(self.state.clip_limits_relative, [0., 1., 0., 1., 0., 1.])

    def test_clip_limits_relative_half_range(self):
        data = Data(label='cube')
        data['x'] = np.arange(120).reshape((3, 4, 10))
        self.state.layers.append(MockLayerState(data))
        pixel_ids = data.pixel_component_ids
        self.state.x_att = pixel_ids[2]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[0]

        self.state.set_limits(0, 5, 0, 2, 0, 1.5)
        assert_allclose(self.state.clip_limits_relative, [0., 0.5, 0., 0.5, 0., 0.5])

    def test_serialization(self):
        self.state.downsample = False
        self.state.resolution = 256
        self.state.x_min = -5
        self.state.x_max = 5
        self.state.visible_axes = False
        self.state.perspective_view = True

        new_state = clone(self.state)

        assert new_state.downsample is False
        assert new_state.resolution == 256
        assert new_state.x_min == -5
        assert new_state.x_max == 5
        assert new_state.visible_axes is False
        assert new_state.perspective_view is True
