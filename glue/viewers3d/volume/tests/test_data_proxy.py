import numpy as np

from glue.core import Data, DataCollection
from glue.core.link_helpers import LinkSame

from ..data_proxy import DataProxy
from ..viewer_state import VolumeViewerState3D


class MockLayerArtist:
    """Mock layer artist for testing DataProxy."""

    def __init__(self, layer):
        self.layer = layer
        self._disabled = False
        self._disable_message = None

    def disable(self, message=None):
        self._disabled = True
        self._disable_message = message

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
