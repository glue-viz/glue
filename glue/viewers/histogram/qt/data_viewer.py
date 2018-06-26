from __future__ import absolute_import, division, print_function

from glue.core.util import update_ticks
from glue.core.roi import RangeROI
from glue.core.subset import roi_to_subset_state
from glue.utils import mpl_to_datetime64, defer_draw, decorate_all_methods

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.histogram.qt.layer_style_editor import HistogramLayerStyleEditor
from glue.viewers.histogram.layer_artist import HistogramLayerArtist
from glue.viewers.histogram.qt.options_widget import HistogramOptionsWidget
from glue.viewers.histogram.state import HistogramViewerState
from glue.viewers.histogram.compat import update_histogram_viewer_state

__all__ = ['HistogramViewer']


@decorate_all_methods(defer_draw)
class HistogramViewer(MatplotlibDataViewer):

    LABEL = '1D Histogram'
    _layer_style_widget_cls = HistogramLayerStyleEditor
    _state_cls = HistogramViewerState
    _options_cls = HistogramOptionsWidget
    _data_artist_cls = HistogramLayerArtist
    _subset_artist_cls = HistogramLayerArtist

    large_data_size = 2e7

    tools = ['select:xrange']

    def __init__(self, session, parent=None, state=None):
        super(HistogramViewer, self).__init__(session, parent, state=state)
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('x_log', self._update_axes)
        self.state.add_callback('normalize', self._update_axes)

    def _update_axes(self, *args):

        if self.state.x_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'x', self.state.x_kinds, self.state.x_log, self.state.x_categories)

            if self.state.x_log:
                self.state.x_axislabel = 'Log ' + self.state.x_att.label
            else:
                self.state.x_axislabel = self.state.x_att.label

        if self.state.normalize:
            self.state.y_axislabel = 'Normalized number'
        else:
            self.state.y_axislabel = 'Number'

        self.axes.figure.canvas.draw()

    def apply_roi(self, roi, override_mode=None):

        if len(self.layers) == 0:  # Force redraw to get rid of ROI
            return self.redraw()

        x_date = 'datetime' in self.state.x_kinds

        if x_date:
            roi = roi.transformed(xfunc=mpl_to_datetime64 if x_date else None)

        bins = self.state.bins

        x = roi.to_polygon()[0]
        lo, hi = min(x), max(x)

        if lo >= bins.min():
            lo = bins[bins <= lo].max()
        if hi <= bins.max():
            hi = bins[bins >= hi].min()

        roi_new = RangeROI(min=lo, max=hi, orientation='x')

        subset_state = roi_to_subset_state(roi_new, x_att=self.state.x_att,
                                           x_categories=self.state.x_categories)

        self.apply_subset_state(subset_state, override_mode=override_mode)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_histogram_viewer_state(rec, context)
