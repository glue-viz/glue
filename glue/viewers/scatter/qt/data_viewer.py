from __future__ import absolute_import, division, print_function

from glue.core.subset import roi_to_subset_state
from glue.core.util import update_ticks

from glue.utils import mpl_to_datetime64, defer_draw, decorate_all_methods
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import ScatterLayerStyleEditor
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.scatter.qt.options_widget import ScatterOptionsWidget
from glue.viewers.scatter.state import ScatterViewerState
from glue.viewers.scatter.compat import update_scatter_viewer_state

__all__ = ['ScatterViewer']


@decorate_all_methods(defer_draw)
class ScatterViewer(MatplotlibDataViewer):

    LABEL = '2D Scatter'
    _layer_style_widget_cls = ScatterLayerStyleEditor
    _state_cls = ScatterViewerState
    _options_cls = ScatterOptionsWidget
    _data_artist_cls = ScatterLayerArtist
    _subset_artist_cls = ScatterLayerArtist

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon']

    def __init__(self, session, parent=None, state=None):
        super(ScatterViewer, self).__init__(session, parent, state=state)
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('y_att', self._update_axes)
        self.state.add_callback('x_log', self._update_axes)
        self.state.add_callback('y_log', self._update_axes)
        self._update_axes()

    def _update_axes(self, *args):

        if self.state.x_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'x', self.state.x_kinds, self.state.x_log, self.state.x_categories)

            if self.state.x_log:
                self.state.x_axislabel = 'Log ' + self.state.x_att.label
            else:
                self.state.x_axislabel = self.state.x_att.label

        if self.state.y_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'y', self.state.y_kinds, self.state.y_log, self.state.y_categories)

            if self.state.y_log:
                self.state.y_axislabel = 'Log ' + self.state.y_att.label
            else:
                self.state.y_axislabel = self.state.y_att.label

        self.axes.figure.canvas.draw()

    @defer_draw
    def apply_roi(self, roi, override_mode=None):

        # Force redraw to get rid of ROI. We do this because applying the
        # subset state below might end up not having an effect on the viewer,
        # for example there may not be any layers, or the active subset may not
        # be one of the layers. So we just explicitly redraw here to make sure
        # a redraw will happen after this method is called.
        self.redraw()

        if len(self.layers) == 0:
            return

        x_date = 'datetime' in self.state.x_kinds
        y_date = 'datetime' in self.state.y_kinds

        if x_date or y_date:
            roi = roi.transformed(xfunc=mpl_to_datetime64 if x_date else None,
                                  yfunc=mpl_to_datetime64 if y_date else None)

        subset_state = roi_to_subset_state(roi,
                                           x_att=self.state.x_att, x_categories=self.state.x_categories,
                                           y_att=self.state.y_att, y_categories=self.state.y_categories)

        self.apply_subset_state(subset_state, override_mode=override_mode)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_scatter_viewer_state(rec, context)
