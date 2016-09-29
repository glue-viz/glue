from __future__ import absolute_import, division, print_function

from glue.external.echo import add_callback
from glue.utils import nonpartial
from glue.viewers.common.qt.mpl_toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core import Data

from glue.viewers.common.qt.mpl_data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import Generic2DLayerStyleEditor
from glue.viewers.scatter.qt.options_widget import ScatterOptionsWidget
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.scatter.state import ScatterViewerState

__all__ = ['ScatterViewer']


class ScatterViewer(MatplotlibDataViewer):

    LABEL = 'New scatter viewer'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = Generic2DLayerStyleEditor
    _state_cls = ScatterViewerState
    _options_cls = ScatterOptionsWidget
    _data_artist_cls = ScatterLayerArtist
    _subset_artist_cls = ScatterLayerArtist

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon']

    def __init__(self, session, parent=None):

        super(ScatterViewer, self).__init__(session, parent)

        add_callback(self.viewer_state, 'log_x', nonpartial(self.update_log))
        add_callback(self.viewer_state, 'log_y', nonpartial(self.update_log))

        add_callback(self.viewer_state, 'xatt', nonpartial(self.update_labels))
        add_callback(self.viewer_state, 'yatt', nonpartial(self.update_labels))

    def update_labels(self):
        if self.viewer_state.xatt is not None:
            self.axes.set_xlabel(self.viewer_state.xatt[0])
        if self.viewer_state.yatt is not None:
            self.axes.set_ylabel(self.viewer_state.yatt[0])

    def update_log(self):
        self.axes.set_xscale('log' if self.viewer_state.log_x else 'linear')
        self.axes.set_yscale('log' if self.viewer_state.log_y else 'linear')
        self.axes.figure.canvas.draw()

    def apply_roi(self, roi):

        # TODO: add back command stack here?
        # cmd = command.ApplyROI(client=self.client, roi=roi)
        # self._session.command_stack.do(cmd)

        # Does subset get applied to all data or just visible data?

        for layer_artist in self._layer_artist_container:

            if not isinstance(layer_artist.layer, Data):
                continue

            x_comp = layer_artist.layer.get_component(self.viewer_state.xatt[0])
            y_comp = layer_artist.layer.get_component(self.viewer_state.yatt[0])

            subset_state = x_comp.subset_from_roi(self.viewer_state.xatt[0], roi,
                                                  other_comp=y_comp,
                                                  other_att=self.viewer_state.yatt[0],
                                                  coord='x')

            mode = EditSubsetMode()
            mode.update(self._data, subset_state, focus_data=layer_artist.layer)
