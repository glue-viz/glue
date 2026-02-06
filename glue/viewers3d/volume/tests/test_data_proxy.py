import numpy as np
from numpy.testing import assert_array_equal

from glue.core import Data, DataCollection
from glue.core.link_helpers import LinkSame

from ..data_proxy import DataProxy
from ..viewer_state import VolumeViewerState3D


class MockLayerArtistState:
    """Mock state for layer artist."""

    def __init__(self, attribute=None):
        self.attribute = attribute


class MockLayerArtist:
    """Mock layer artist for testing DataProxy."""

    _id_counter = 0

    def __init__(self, layer, attribute=None):
        self.layer = layer
        self._disabled = False
        self._disable_message = None
        self.state = MockLayerArtistState(attribute)
        MockLayerArtist._id_counter += 1
        self.id = MockLayerArtist._id_counter

    def disable(self, message=None):
        self._disabled = True
        self._disable_message = message

    def disable_incompatible_subset(self):
        self._disabled = True
        self._disable_message = 'Incompatible subset'

    def enable(self):
        self._disabled = False
        self._disable_message = None


class MockLayerState:
    """Simple mock layer state for testing viewer state."""

    def __init__(self, layer):
        self.layer = layer


class TestDataProxy:

    def setup_method(self, method):
        self.data_4d = Data(label='data_4d')
        self.data_4d['x'] = np.arange(5 * 4 * 2 * 7).reshape((5, 4, 2, 7))

        self.data_4d_2 = Data(label='data_4d_2')
        self.data_4d_2['x'] = np.arange(6 * 3 * 7 * 11).reshape((6, 3, 7, 11))

        self.dc = DataCollection([self.data_4d, self.data_4d_2])

        self.dc.add_link(LinkSame(self.data_4d.pixel_component_ids[0],
                                  self.data_4d_2.pixel_component_ids[3]))
        self.dc.add_link(LinkSame(self.data_4d.pixel_component_ids[1],
                                  self.data_4d_2.pixel_component_ids[0]))
        self.dc.add_link(LinkSame(self.data_4d.pixel_component_ids[2],
                                  self.data_4d_2.pixel_component_ids[1]))

        self.state = VolumeViewerState3D()
        self.state.layers.append(MockLayerState(self.data_4d))

    def test_shape_with_reference_data(self):
        layer_artist = MockLayerArtist(self.data_4d)

        self.state.x_att = self.data_4d.pixel_component_ids[0]
        self.state.y_att = self.data_4d.pixel_component_ids[1]
        self.state.z_att = self.data_4d.pixel_component_ids[2]

        proxy = DataProxy(self.state, layer_artist)
        assert proxy.shape == (2, 4, 5)

    def test_shape_updates_when_x_att_changes(self):
        layer_artist = MockLayerArtist(self.data_4d)

        self.state.x_att = self.data_4d.pixel_component_ids[0]
        self.state.y_att = self.data_4d.pixel_component_ids[1]
        self.state.z_att = self.data_4d.pixel_component_ids[2]

        proxy = DataProxy(self.state, layer_artist)
        assert proxy.shape == (2, 4, 5)

        self.state.x_att = self.data_4d.pixel_component_ids[3]
        assert proxy.shape == (2, 4, 7)

    def test_shape_with_permuted_axes(self):
        layer_artist = MockLayerArtist(self.data_4d)

        self.state.x_att = self.data_4d.pixel_component_ids[2]
        self.state.y_att = self.data_4d.pixel_component_ids[1]
        self.state.z_att = self.data_4d.pixel_component_ids[3]

        proxy = DataProxy(self.state, layer_artist)
        assert proxy.shape == (7, 4, 2)

    def test_shape_linked_data_no_z_link(self):
        self.state.layers.append(MockLayerState(self.data_4d_2))
        layer_artist_2 = MockLayerArtist(self.data_4d_2)

        # Use z_att = pixel[3] which has no link to data_4d_2
        # Links are: data_4d[0]<->data_4d_2[3], data_4d[1]<->data_4d_2[0], data_4d[2]<->data_4d_2[1]
        # data_4d[3] is NOT linked
        self.state.x_att = self.data_4d.pixel_component_ids[2]
        self.state.y_att = self.data_4d.pixel_component_ids[1]
        self.state.z_att = self.data_4d.pixel_component_ids[3]

        proxy2 = DataProxy(self.state, layer_artist_2)
        # data_4d[3] has no link to data_4d_2, so shape is (0, 0, 0)
        assert proxy2.shape == (0, 0, 0)
        assert layer_artist_2._disabled

    def test_shape_linked_data_with_valid_links(self):
        self.state.layers.append(MockLayerState(self.data_4d_2))
        layer_artist_2 = MockLayerArtist(self.data_4d_2)

        # Use axes that are all linked between the two datasets
        # data_4d[0] <-> data_4d_2[3], data_4d[1] <-> data_4d_2[0], data_4d[2] <-> data_4d_2[1]
        self.state.x_att = self.data_4d.pixel_component_ids[0]
        self.state.y_att = self.data_4d.pixel_component_ids[1]
        self.state.z_att = self.data_4d.pixel_component_ids[2]

        proxy2 = DataProxy(self.state, layer_artist_2)
        # data_4d_2 shape is (6, 3, 7, 11)
        # x_att -> data_4d[0] -> data_4d_2[3] -> size 11
        # y_att -> data_4d[1] -> data_4d_2[0] -> size 6
        # z_att -> data_4d[2] -> data_4d_2[1] -> size 3
        assert proxy2.shape == (3, 6, 11)

    def test_shape_linked_data_axes_permuted(self):
        self.state.layers.append(MockLayerState(self.data_4d_2))

        # Add the missing link for full linking
        self.dc.add_link(LinkSame(self.data_4d.pixel_component_ids[3],
                                  self.data_4d_2.pixel_component_ids[2]))

        layer_artist_2 = MockLayerArtist(self.data_4d_2)

        self.state.x_att = self.data_4d.pixel_component_ids[2]
        self.state.y_att = self.data_4d.pixel_component_ids[0]
        self.state.z_att = self.data_4d.pixel_component_ids[1]

        proxy2 = DataProxy(self.state, layer_artist_2)
        # x_att -> data_4d[2] -> data_4d_2[1] -> size 3
        # y_att -> data_4d[0] -> data_4d_2[3] -> size 11
        # z_att -> data_4d[1] -> data_4d_2[0] -> size 6
        assert proxy2.shape == (6, 11, 3)

    def test_shape_linked_data_y_att_change(self):
        self.state.layers.append(MockLayerState(self.data_4d_2))

        self.dc.add_link(LinkSame(self.data_4d.pixel_component_ids[3],
                                  self.data_4d_2.pixel_component_ids[2]))

        layer_artist_2 = MockLayerArtist(self.data_4d_2)

        self.state.x_att = self.data_4d.pixel_component_ids[2]
        self.state.y_att = self.data_4d.pixel_component_ids[0]
        self.state.z_att = self.data_4d.pixel_component_ids[1]

        proxy2 = DataProxy(self.state, layer_artist_2)
        assert proxy2.shape == (6, 11, 3)

        self.state.y_att = self.data_4d.pixel_component_ids[3]
        # y_att -> data_4d[3] -> data_4d_2[2] -> size 7
        assert proxy2.shape == (6, 7, 3)

    def test_subset_shape(self):
        subset = self.data_4d.new_subset()
        subset.subset_state = self.data_4d.id['x'] > 10

        layer_artist = MockLayerArtist(subset)

        self.state.x_att = self.data_4d.pixel_component_ids[0]
        self.state.y_att = self.data_4d.pixel_component_ids[1]
        self.state.z_att = self.data_4d.pixel_component_ids[2]

        proxy = DataProxy(self.state, layer_artist)
        # Subset uses parent data shape
        assert proxy.shape == (2, 4, 5)


class TestDataProxyComputeBuffer:
    """Tests for DataProxy.compute_fixed_resolution_buffer()"""

    def setup_method(self, method):
        # Create simple 3D data with known values
        self.data_3d = Data(label='cube_3d')
        self.data_3d['x'] = np.arange(60).reshape((3, 4, 5))

        self.dc = DataCollection([self.data_3d])

        self.state = VolumeViewerState3D()
        self.state.layers.append(MockLayerState(self.data_3d))

        # Set up axes in natural order: x=axis2, y=axis1, z=axis0
        self.state.x_att = self.data_3d.pixel_component_ids[2]
        self.state.y_att = self.data_3d.pixel_component_ids[1]
        self.state.z_att = self.data_3d.pixel_component_ids[0]

    def test_compute_buffer_basic(self):
        layer_artist = MockLayerArtist(self.data_3d,
                                       attribute=self.data_3d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        # Request full resolution buffer
        # bounds format: [(z_min, z_max, z_steps), (y_min, y_max, y_steps), (x_min, x_max, x_steps)]
        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (3, 4, 5)
        assert_array_equal(result, self.data_3d['x'])

    def test_compute_buffer_subregion(self):
        layer_artist = MockLayerArtist(self.data_3d,
                                       attribute=self.data_3d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        # Request only a portion of the data
        bounds = [(0, 1, 2), (0, 1, 2), (0, 1, 2)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (2, 2, 2)
        # Should match the corner of the original data
        assert_array_equal(result, self.data_3d['x'][0:2, 0:2, 0:2])

    def test_compute_buffer_enables_layer(self):
        layer_artist = MockLayerArtist(self.data_3d,
                                       attribute=self.data_3d.id['x'])
        layer_artist._disabled = True

        proxy = DataProxy(self.state, layer_artist)

        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        proxy.compute_fixed_resolution_buffer(bounds)

        assert not layer_artist._disabled

    def test_compute_buffer_subset(self):
        subset = self.data_3d.new_subset()
        subset.subset_state = self.data_3d.id['x'] > 30

        layer_artist = MockLayerArtist(subset)

        proxy = DataProxy(self.state, layer_artist)

        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (3, 4, 5)
        # Subset mask: True where x > 30
        expected_mask = self.data_3d['x'] > 30
        assert_array_equal(result, expected_mask)

    def test_compute_buffer_subset_enables_layer(self):
        subset = self.data_3d.new_subset()
        subset.subset_state = self.data_3d.id['x'] > 30

        layer_artist = MockLayerArtist(subset)
        layer_artist._disabled = True

        proxy = DataProxy(self.state, layer_artist)

        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        proxy.compute_fixed_resolution_buffer(bounds)

        assert not layer_artist._disabled

    def test_compute_buffer_with_permuted_axes(self):
        # Set up axes in non-natural order
        self.state.x_att = self.data_3d.pixel_component_ids[0]
        self.state.y_att = self.data_3d.pixel_component_ids[2]
        self.state.z_att = self.data_3d.pixel_component_ids[1]

        layer_artist = MockLayerArtist(self.data_3d,
                                       attribute=self.data_3d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        # Shape should now be (y_size, x_size, z_size) = (4, 5, 3) based on new axis mapping
        # z_att=axis1 -> size 4, y_att=axis2 -> size 5, x_att=axis0 -> size 3
        bounds = [(0, 3, 4), (0, 4, 5), (0, 2, 3)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (4, 5, 3)

    def test_compute_buffer_returns_zeros_when_layer_artist_none(self):
        layer_artist = MockLayerArtist(self.data_3d,
                                       attribute=self.data_3d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        # Simulate weakref being garbage collected
        proxy._layer_artist = lambda: None

        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (3, 4, 5)
        assert np.all(result == 0)

    def test_compute_buffer_returns_zeros_when_viewer_state_none(self):
        layer_artist = MockLayerArtist(self.data_3d,
                                       attribute=self.data_3d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        # Simulate weakref being garbage collected
        proxy._viewer_state = lambda: None

        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (3, 4, 5)
        assert np.all(result == 0)


class TestDataProxyComputeBuffer4D:
    """Tests for compute_fixed_resolution_buffer with 4D data."""

    def setup_method(self, method):
        self.data_4d = Data(label='cube_4d')
        self.data_4d['x'] = np.arange(120).reshape((2, 3, 4, 5))

        self.dc = DataCollection([self.data_4d])

        self.state = VolumeViewerState3D()
        self.state.layers.append(MockLayerState(self.data_4d))

        # Use last 3 axes for display
        self.state.x_att = self.data_4d.pixel_component_ids[3]
        self.state.y_att = self.data_4d.pixel_component_ids[2]
        self.state.z_att = self.data_4d.pixel_component_ids[1]

        # Set slice for first axis
        self.state.slices = (0, 0, 0, 0)

    def test_compute_buffer_4d_first_slice(self):
        layer_artist = MockLayerArtist(self.data_4d,
                                       attribute=self.data_4d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        # Request full 3D slice
        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (3, 4, 5)
        # Should get first slice along axis 0
        assert_array_equal(result, self.data_4d['x'][0, :, :, :])

    def test_compute_buffer_4d_second_slice(self):
        self.state.slices = (1, 0, 0, 0)

        layer_artist = MockLayerArtist(self.data_4d,
                                       attribute=self.data_4d.id['x'])

        proxy = DataProxy(self.state, layer_artist)

        bounds = [(0, 2, 3), (0, 3, 4), (0, 4, 5)]
        result = proxy.compute_fixed_resolution_buffer(bounds)

        assert result.shape == (3, 4, 5)
        # Should get second slice along axis 0
        assert_array_equal(result, self.data_4d['x'][1, :, :, :])
