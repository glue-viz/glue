from __future__ import absolute_import, division, print_function

from glue.utils import nonpartial
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core import Data
from glue.core.util import update_ticks
from glue.core.roi import RangeROI
from glue.utils import defer_draw

from glue.viewers.common.qt.mpl_data_viewer import MatplotlibDataViewer
from glue.viewers.histogram_new.qt.layer_style_editor import HistogramLayerStyleEditor
from glue.viewers.histogram_new.layer_artist import HistogramLayerArtist
from glue.viewers.histogram_new.qt.options_widget import HistogramOptionsWidget
from glue.viewers.histogram_new.state import HistogramViewerState

__all__ = ['HistogramViewer']


class HistogramViewer(MatplotlibDataViewer):

    LABEL = 'New histogram viewer'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = HistogramLayerStyleEditor
    _state_cls = HistogramViewerState
    _options_cls = HistogramOptionsWidget
    _data_artist_cls = HistogramLayerArtist
    _subset_artist_cls = HistogramLayerArtist

    tools = ['select:xrange']

    def __init__(self, session, parent=None):
        super(HistogramViewer, self).__init__(session, parent)
        self.viewer_state.add_callback('xatt', nonpartial(self._update_axes))
        self.viewer_state.add_callback('log_x', nonpartial(self._update_axes))
        self.viewer_state.add_callback('normalize', nonpartial(self._update_axes))

    @defer_draw
    def _update_axes(self):

        if self.viewer_state.xatt is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'x', self.viewer_state._get_x_components(), False)

            if self.viewer_state.log_x:
                self.axes.set_xlabel('Log ' + self.viewer_state.xatt.label)
            else:
                self.axes.set_xlabel(self.viewer_state.xatt.label)

        if self.viewer_state.normalize:
            self.axes.set_ylabel('Normalized number')
        else:
            self.axes.set_ylabel('Number')

        self.axes.figure.canvas.draw()

    def apply_roi(self, roi):

        # TODO: move this to state class?

        # TODO: add back command stack here so as to be able to undo?
        # cmd = command.ApplyROI(client=self.client, roi=roi)
        # self._session.command_stack.do(cmd)

        # TODO Does subset get applied to all data or just visible data?

        # Expand roi to match bin edges
        # TODO: make this an option

        bins = self.viewer_state.bins

        x = roi.to_polygon()[0]
        lo, hi = min(x), max(x)

        if lo >= bins.min():
            lo = bins[bins <= lo].max()
        if hi <= bins.max():
            hi = bins[bins >= hi].min()

        roi_new = RangeROI(min=lo, max=hi, orientation='x')

        for layer_artist in self._layer_artist_container:

            if not isinstance(layer_artist.layer, Data):
                continue

            x_comp = layer_artist.layer.get_component(self.viewer_state.xatt)

            subset_state = x_comp.subset_from_roi(self.viewer_state.xatt, roi_new,
                                                  coord='x')

            mode = EditSubsetMode()
            mode.update(self._data, subset_state, focus_data=layer_artist.layer)
