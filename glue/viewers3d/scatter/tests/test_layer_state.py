from glue.core import Data, DataCollection
from glue.core.tests.test_state import clone

from ..layer_state import ScatterLayerState


class TestScatterLayerState:

    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3, 4, 5], label='test_data')
        self.data_collection = DataCollection([self.data])
        self.state = ScatterLayerState(layer=self.data)

    def test_size_syncs_to_layer_markersize(self):
        self.state.size = 15
        assert self.data.style.markersize == 15

    def test_layer_markersize_syncs_to_size(self):
        self.data.style.markersize = 20
        assert self.state.size == 20

    def test_flip_size(self):
        self.state.size_attribute = self.data.id['x']
        vmin_before = self.state.size_vmin
        vmax_before = self.state.size_vmax
        self.state.flip_size()
        assert self.state.size_vmin == vmax_before
        assert self.state.size_vmax == vmin_before

    def test_flip_cmap(self):
        self.state.cmap_attribute = self.data.id['x']
        vmin_before = self.state.cmap_vmin
        vmax_before = self.state.cmap_vmax
        self.state.flip_cmap()
        assert self.state.cmap_vmin == vmax_before
        assert self.state.cmap_vmax == vmin_before

    def test_serialization(self):
        self.state.color = '#abcdef'
        self.state.alpha = 0.7
        self.state.size = 12
        self.state.size_mode = 'Linear'
        self.state.size_scaling = 2.5
        self.state.color_mode = 'Linear'
        self.state.xerr_visible = True
        self.state.yerr_visible = True
        self.state.zerr_visible = True
        self.state.vector_visible = True
        self.state.vector_scaling = 1.5
        self.state.vector_origin = 'middle'
        self.state.vector_arrowhead = True

        new_state = clone(self.state)

        assert new_state.color == '#abcdef'
        assert new_state.alpha == 0.7
        assert new_state.size == 12
        assert new_state.size_mode == 'Linear'
        assert new_state.size_scaling == 2.5
        assert new_state.color_mode == 'Linear'
        assert new_state.xerr_visible is True
        assert new_state.yerr_visible is True
        assert new_state.zerr_visible is True
        assert new_state.vector_visible is True
        assert new_state.vector_scaling == 1.5
        assert new_state.vector_origin == 'middle'
        assert new_state.vector_arrowhead is True
