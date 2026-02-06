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

    def test_numpy_slice_permutation_no_reference_data(self):
        slices, perm = self.state.numpy_slice_permutation
        assert slices is None
        assert perm is None

    def test_numpy_slice_permutation_3d_default_axes(self):
        data = Data(label='cube')
        data['x'] = np.arange(24).reshape((2, 3, 4))
        self.state.layers.append(MockLayerState(data))
        pixel_ids = data.pixel_component_ids
        # Set axes: x=axis2, y=axis1, z=axis0
        self.state.x_att = pixel_ids[2]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[0]

        slices, perm = self.state.numpy_slice_permutation
        # All 3 axes are coordinate axes, so all should be slice(None)
        assert slices == [slice(None), slice(None), slice(None)]
        # With x=2, y=1, z=0, axes are already in order so perm should be identity
        assert perm == [0, 1, 2]

    def test_numpy_slice_permutation_swapped_axes(self):
        data = Data(label='cube')
        data['x'] = np.arange(24).reshape((2, 3, 4))
        self.state.layers.append(MockLayerState(data))
        pixel_ids = data.pixel_component_ids
        # Set axes: x=axis0, y=axis1, z=axis2 (reversed from typical)
        self.state.x_att = pixel_ids[0]
        self.state.y_att = pixel_ids[1]
        self.state.z_att = pixel_ids[2]

        slices, perm = self.state.numpy_slice_permutation
        assert slices == [slice(None), slice(None), slice(None)]
        # With x=0, y=1, z=2, need to reverse the order
        assert perm == [2, 1, 0]
