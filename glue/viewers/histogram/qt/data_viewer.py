from __future__ import absolute_import, division, print_function

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.util import update_ticks
from glue.core.roi import RangeROI
from glue.core import command

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.histogram.qt.layer_style_editor import HistogramLayerStyleEditor
from glue.viewers.histogram.layer_artist import HistogramLayerArtist
from glue.viewers.histogram.qt.options_widget import HistogramOptionsWidget
from glue.viewers.histogram.state import HistogramViewerState
from glue.viewers.histogram.compat import update_histogram_viewer_state

__all__ = ['HistogramViewer']


class HistogramViewer(MatplotlibDataViewer):

    LABEL = '1D Histogram'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = HistogramLayerStyleEditor
    _state_cls = HistogramViewerState
    _options_cls = HistogramOptionsWidget
    _data_artist_cls = HistogramLayerArtist
    _subset_artist_cls = HistogramLayerArtist

    tools = ['select:xrange']

    def __init__(self, session, parent=None, state=None):
        super(HistogramViewer, self).__init__(session, parent, state=state)
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('x_log', self._update_axes)
        self.state.add_callback('normalize', self._update_axes)

    def _update_axes(self, *args):

        if self.state.x_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'x', self.state._get_x_components(), False)

            if self.state.x_log:
                self.axes.set_xlabel('Log ' + self.state.x_att.label)
            else:
                self.axes.set_xlabel(self.state.x_att.label)

        if self.state.normalize:
            self.axes.set_ylabel('Normalized number')
        else:
            self.axes.set_ylabel('Number')

        self.axes.figure.canvas.draw()

    # TODO: move some of the ROI stuff to state class?

    def apply_roi(self, roi):
        if len(self.layers) > 0:
            cmd = command.ApplyROI(data_collection=self._data,
                                   roi=roi, apply_func=self._apply_roi)
            self._session.command_stack.do(cmd)
        else:
            # Make sure we force a redraw to get rid of the ROI
            self.axes.figure.canvas.draw()

    def _apply_roi(self, roi):

        # TODO Does subset get applied to all data or just visible data?

        bins = self.state.bins

        x = roi.to_polygon()[0]
        lo, hi = min(x), max(x)

        if lo >= bins.min():
            lo = bins[bins <= lo].max()
        if hi <= bins.max():
            hi = bins[bins >= hi].min()

        roi_new = RangeROI(min=lo, max=hi, orientation='x')

        x_comp = self.state.x_att.parent.get_component(self.state.x_att)

        subset_state = x_comp.subset_from_roi(self.state.x_att, roi_new,
                                              coord='x')

        mode = EditSubsetMode()
        mode.update(self._data, subset_state)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_histogram_viewer_state(rec, context)
