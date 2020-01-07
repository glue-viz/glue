from glue.utils import defer_draw, decorate_all_methods
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.histogram.qt.layer_style_editor import HistogramLayerStyleEditor
from glue.viewers.histogram.qt.options_widget import HistogramOptionsWidget
from glue.viewers.histogram.qt.layer_artist import QThreadedHistogramLayerArtist
from glue.viewers.histogram.state import HistogramViewerState

from glue.viewers.histogram.viewer import MatplotlibHistogramMixin

__all__ = ['HistogramViewer']


@decorate_all_methods(defer_draw)
class HistogramViewer(MatplotlibHistogramMixin, MatplotlibDataViewer):

    LABEL = '1D Histogram'

    _layer_style_widget_cls = HistogramLayerStyleEditor
    _options_cls = HistogramOptionsWidget

    _state_cls = HistogramViewerState
    _data_artist_cls = QThreadedHistogramLayerArtist
    _subset_artist_cls = QThreadedHistogramLayerArtist

    large_data_size = 2e7

    tools = ['select:xrange']

    def __init__(self, session, parent=None, state=None):
        super(HistogramViewer, self).__init__(session, parent=parent, state=state)
        MatplotlibHistogramMixin.setup_callbacks(self)
