from ..state import MatplotlibLegendState, MatplotlibDataViewerState
from glue.config import settings
from glue.core.tests.test_state import clone

from matplotlib.colors import to_rgba


class TestMatplotlibDataViewerState:
    def setup_method(self, method):
        self.state = MatplotlibDataViewerState()

    def test_legend_serialization(self):
        legend_state = self.state.legend
        legend_state.visible = True
        legend_state.location = "best"
        legend_state.title = "Legend"
        legend_state.fontsize = 13
        legend_state.alpha = 0.7
        legend_state.frame_color = "#1e00f1"
        legend_state.show_edge = False
        legend_state.text_color = "#fad8f1"

        new_state = clone(self.state)
        new_legend_state = new_state.legend
        assert new_legend_state.visible
        assert new_legend_state.location == "best"
        assert new_legend_state.title == "Legend"
        assert new_legend_state.fontsize == 13
        assert new_legend_state.alpha == 0.7
        assert new_legend_state.frame_color == "#1e00f1"
        assert not new_legend_state.show_edge
        assert new_legend_state.text_color == "#fad8f1"


class TestMatplotlibLegendState:
    def setup_method(self, method):
        self.state = MatplotlibLegendState()

    def test_draggable(self):
        self.state.location = 'draggable'
        assert self.state.draggable
        assert self.state.mpl_location == 'best'

    def test_no_draggable(self):
        self.state.location = 'lower left'
        assert not self.state.draggable
        assert self.state.mpl_location == 'lower left'

    def test_no_edge(self):
        self.state.show_edge = False
        assert self.state.edge_color is None

    def test_default_color(self):
        assert self.state.frame_color == settings.BACKGROUND_COLOR
        assert self.state.edge_color == to_rgba(settings.FOREGROUND_COLOR, self.state.alpha)
