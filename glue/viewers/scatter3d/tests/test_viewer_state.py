from glue.core import Data
from glue.core.tests.test_state import clone

from ..viewer_state import ScatterViewerState3D
from ..layer_state import ScatterLayerState


def get_choice_labels(choices):
    """Extract labels from choices, filtering out separators."""
    return [c.label for c in choices if hasattr(c, 'label')]


class TestScatterViewerState3D:

    def setup_method(self, method):
        self.data1 = Data(x=[1, 2, 3], label='data1')
        self.data2 = Data(a=[4, 5, 6], b=[7, 8, 9], label='data2')
        self.state = ScatterViewerState3D()

    def test_adding_layer_populates_att_helpers(self):
        layer_state = ScatterLayerState(layer=self.data1)
        self.state.layers.append(layer_state)

        choices = set(get_choice_labels(self.state.x_att_helper.choices))
        assert 'x' in choices

    def test_adding_second_layer_adds_its_components(self):
        layer_state1 = ScatterLayerState(layer=self.data1)
        layer_state2 = ScatterLayerState(layer=self.data2)
        self.state.layers.append(layer_state1)
        self.state.layers.append(layer_state2)

        choices = set(get_choice_labels(self.state.x_att_helper.choices))
        assert 'x' in choices
        assert 'a' in choices
        assert 'b' in choices

    def test_removing_layer_updates_att_helpers(self):
        layer_state1 = ScatterLayerState(layer=self.data1)
        layer_state2 = ScatterLayerState(layer=self.data2)
        self.state.layers.append(layer_state1)
        self.state.layers.append(layer_state2)

        self.state.layers.remove(layer_state2)

        choices = set(get_choice_labels(self.state.x_att_helper.choices))
        assert 'x' in choices

    def test_serialization(self):
        self.state.x_min = -5
        self.state.x_max = 5
        self.state.y_min = -10
        self.state.y_max = 10
        self.state.z_min = -15
        self.state.z_max = 15
        self.state.visible_axes = False
        self.state.native_aspect = True

        new_state = clone(self.state)

        assert new_state.x_min == -5
        assert new_state.x_max == 5
        assert new_state.y_min == -10
        assert new_state.y_max == 10
        assert new_state.z_min == -15
        assert new_state.z_max == 15
        assert new_state.visible_axes is False
        assert new_state.native_aspect is True
