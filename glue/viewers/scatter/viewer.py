from glue.core.subset import roi_to_subset_state
from glue.core.util import update_ticks
from glue.core import glue_pickle as gp

from glue.utils import mpl_to_datetime64
from glue.viewers.scatter.compat import update_scatter_viewer_state
from glue.viewers.matplotlib.mpl_axes import init_mpl
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
import numpy as np


__all__ = ['MatplotlibScatterMixin']


class MplProjectionTransform(object):
    def __init__(self, axes):
        self._transform = (axes.transData + axes.transAxes.inverted()).frozen() if axes else None

    def __call__(self, x,y):
        assert self._transform is not None
        assert x.shape == y.shape
        points = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        res = self._transform.transform(points)
        out = np.hsplit(res,2)
        return out[0].reshape(x.shape), out[1].reshape(y.shape)

    def __gluestate__(self, context):
        return dict(transform=context.id(self._transform))

    @classmethod
    def __setgluestate__(cls, rec, context):
        obj = cls()
        obj._transform = context.object(rec['transform'])
        return obj

class MatplotlibScatterMixin(object):

    def setup_callbacks(self):
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('y_att', self._update_axes)
        self.state.add_callback('x_log', self._update_axes)
        self.state.add_callback('y_log', self._update_axes)
        self.state.add_callback('plot_mode', self._update_projection)
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

        self.axes.figure.canvas.draw_idle()

    def _update_projection(self, *args):

        old_layers = self.layers

        self.figure.delaxes(self.axes)
        _, self.axes = init_mpl(self.figure, projection=self.state.plot_mode)
        for layer in old_layers:
            layer._set_axes(self.axes)
            layer.update()
        self.axes.callbacks.connect('xlim_changed', self.limits_from_mpl)
        self.axes.callbacks.connect('ylim_changed', self.limits_from_mpl)
        self.limits_from_mpl()
        self.removeToolBar(self.toolbar)
        self.initialize_toolbar()
        self.figure.canvas.draw_idle()

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

        use_transform = self.state.plot_mode != 'rectilinear'
        subset_state = roi_to_subset_state(roi,
                                           x_att=self.state.x_att, x_categories=self.state.x_categories,
                                           y_att=self.state.y_att, y_categories=self.state.y_categories,
                                           use_pretransform = use_transform)
        if use_transform:
            subset_state.pretransform = MplProjectionTransform(self.axes)

        self.apply_subset_state(subset_state, override_mode=override_mode)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_scatter_viewer_state(rec, context)
