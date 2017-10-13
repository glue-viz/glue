from __future__ import absolute_import, division, print_function

from glue.core import command
from glue.utils import nonpartial
from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.util import update_ticks

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import GenericLayerStyleEditor
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.scatter.qt.options_widget import ScatterOptionsWidget
from glue.viewers.scatter.state import ScatterViewerState
from glue.viewers.scatter.compat import update_scatter_viewer_state

__all__ = ['ScatterViewer']


class ScatterViewer(MatplotlibDataViewer):

    LABEL = '2D Scatter'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = GenericLayerStyleEditor
    _state_cls = ScatterViewerState
    _options_cls = ScatterOptionsWidget
    _data_artist_cls = ScatterLayerArtist
    _subset_artist_cls = ScatterLayerArtist

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon']

    def __init__(self, session, parent=None, state=None):
        super(ScatterViewer, self).__init__(session, parent, state=state)
        self.state.add_callback('x_att', nonpartial(self._update_axes))
        self.state.add_callback('y_att', nonpartial(self._update_axes))
        self.state.add_callback('x_log', nonpartial(self._update_axes))
        self.state.add_callback('y_log', nonpartial(self._update_axes))

    def _update_axes(self):

        if self.state.x_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'x', self.state._get_x_components(), False)

            if self.state.x_log:
                self.axes.set_xlabel('Log ' + self.state.x_att.label)
            else:
                self.axes.set_xlabel(self.state.x_att.label)

        if self.state.y_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'y', self.state._get_y_components(), False)

            if self.state.y_log:
                self.axes.set_ylabel('Log ' + self.state.y_att.label)
            else:
                self.axes.set_ylabel(self.state.y_att.label)

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

        x_comp = self.state.x_att.parent.get_component(self.state.x_att)
        y_comp = self.state.y_att.parent.get_component(self.state.y_att)

        subset_state = x_comp.subset_from_roi(self.state.x_att, roi,
                                              other_comp=y_comp,
                                              other_att=self.state.y_att,
                                              coord='x')

        mode = EditSubsetMode()
        mode.update(self._data, subset_state)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_scatter_viewer_state(rec, context)
