from ..state import MatplotlibLegendState
from glue.config import settings

from matplotlib.colors import to_rgba


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
