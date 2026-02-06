from glue.core import Data, DataCollection
from glue.core.hub import HubListener
from glue.core.message import LayerArtistUpdatedMessage
from glue.core.tests.test_state import clone

from ..layer_state import LayerState3D


class TestLayerState3D:

    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3], y=[4, 5, 6], label='test_data')
        self.data_collection = DataCollection([self.data])
        self.state = LayerState3D(layer=self.data)

    def test_color_sync_from_state_to_layer(self):
        self.state.color = '#ff0000'
        assert self.data.style.color == '#ff0000'

    def test_color_sync_from_layer_to_state(self):
        self.data.style.color = '#00ff00'
        assert self.state.color == '#00ff00'

    def test_alpha_sync_from_state_to_layer(self):
        self.state.alpha = 0.5
        assert self.data.style.alpha == 0.5

    def test_alpha_sync_from_layer_to_state(self):
        self.data.style.alpha = 0.3
        assert self.state.alpha == 0.3

    def test_state_changes_broadcast_message(self):
        messages_received = []

        class TestListener(HubListener):
            def register_to_hub(self, hub):
                hub.subscribe(self, LayerArtistUpdatedMessage,
                              handler=self.receive_message)

            def receive_message(self, message):
                messages_received.append(message)

        listener = TestListener()
        listener.register_to_hub(self.data_collection.hub)

        self.state.color = '#ff0000'

        assert len(messages_received) >= 1
        assert any(isinstance(msg, LayerArtistUpdatedMessage)
                   for msg in messages_received)

    def test_subset_color_syncs(self):
        subset = self.data.new_subset()
        subset.subset_state = self.data.id['x'] > 1
        state = LayerState3D(layer=subset)

        state.color = '#ff0000'
        assert subset.style.color == '#ff0000'

    def test_subset_alpha_syncs(self):
        subset = self.data.new_subset()
        subset.subset_state = self.data.id['x'] > 1
        state = LayerState3D(layer=subset)

        state.alpha = 0.5
        assert subset.style.alpha == 0.5

    def test_serialization(self):
        self.state.color = '#123456'
        self.state.alpha = 0.3

        new_state = clone(self.state)

        assert new_state.color == '#123456'
        assert new_state.alpha == 0.3
