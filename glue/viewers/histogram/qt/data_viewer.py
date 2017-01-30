from __future__ import absolute_import, division, print_function

from glue.utils import nonpartial
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core import Data

from glue.viewers.common.qt.mpl_data_viewer import MatplotlibDataViewer
from glue.viewers.histogram.qt.layer_style_editor import HistogramLayerStyleEditor
from glue.viewers.histogram.layer_artist import HistogramLayerArtist
from glue.viewers.histogram.qt.options_widget import HistogramOptionsWidget
from glue.viewers.histogram.state import HistogramViewerState

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

        self.viewer_state.add_callback('log_x', nonpartial(self.update_log))
        self.viewer_state.add_callback('log_y', nonpartial(self.update_log))

        self.viewer_state.add_callback('xatt', nonpartial(self.update_labels))

    def update_labels(self):
        if self.viewer_state.xatt is not None:
            self.axes.set_xlabel(self.viewer_state.xatt[0])
        self.axes.set_ylabel('Number')

    def update_log(self):
        self.axes.set_xscale('log' if self.viewer_state.log_x else 'linear')
        self.axes.set_yscale('log' if self.viewer_state.log_y else 'linear')

    def apply_roi(self, roi):

        # TODO: add back command stack here?
        # cmd = command.ApplyROI(client=self.client, roi=roi)
        # self._session.command_stack.do(cmd)

        # Does subset get applied to all data or just visible data?

        for layer_artist in self._layer_artist_container:

            if not isinstance(layer_artist.layer, Data):
                continue

            x_comp = layer_artist.layer.get_component(self.viewer_state.xatt[0])

            subset_state = x_comp.subset_from_roi(self.viewer_state.xatt[0], roi,
                                                  coord='x')

            mode = EditSubsetMode()
            mode.update(self._data, subset_state, focus_data=layer_artist.layer)
