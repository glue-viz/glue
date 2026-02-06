import numpy as np

from glue.core import Data, DataCollection
from glue.core.tests.test_state import clone

from ..layer_state import VolumeLayerState


class TestVolumeLayerState:

    def setup_method(self, method):
        self.data = Data(label='test_cube')
        self.data['x'] = np.arange(24).reshape((2, 3, 4))
        self.data_collection = DataCollection([self.data])
        self.state = VolumeLayerState(layer=self.data)

    def test_flip_limits(self):
        self.state.attribute = self.data.id['x']
        vmin_before = self.state.vmin
        vmax_before = self.state.vmax
        self.state.flip_limits()
        assert self.state.vmin == vmax_before
        assert self.state.vmax == vmin_before

    def test_subset_has_fixed_vmin_vmax(self):
        subset = self.data.new_subset()
        subset.subset_state = self.data.id['x'] > 10
        state = VolumeLayerState(layer=subset)

        assert state.vmin == 0
        assert state.vmax == 1

    def test_serialization(self):
        self.state.color = '#abcdef'
        self.state.alpha = 0.6
        self.state.color_mode = 'Linear'
        self.state.subset_mode = 'outline'

        new_state = clone(self.state)

        assert new_state.color == '#abcdef'
        assert new_state.alpha == 0.6
        assert new_state.color_mode == 'Linear'
        assert new_state.subset_mode == 'outline'
