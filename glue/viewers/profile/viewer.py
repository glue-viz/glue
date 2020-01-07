from glue.core.subset import roi_to_subset_state

__all__ = ['MatplotlibProfileMixin']


class MatplotlibProfileMixin(object):

    def setup_callbacks(self):
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('normalize', self._update_axes)

    def _update_axes(self, *args):

        if self.state.x_att is not None:
            self.state.x_axislabel = self.state.x_att.label

        if self.state.normalize:
            self.state.y_axislabel = 'Normalized data values'
        else:
            self.state.y_axislabel = 'Data values'

        self.axes.figure.canvas.draw_idle()

    def apply_roi(self, roi, override_mode=None):

        # Force redraw to get rid of ROI. We do this because applying the
        # subset state below might end up not having an effect on the viewer,
        # for example there may not be any layers, or the active subset may not
        # be one of the layers. So we just explicitly redraw here to make sure
        # a redraw will happen after this method is called.
        self.redraw()

        if len(self.layers) == 0:
            return

        subset_state = roi_to_subset_state(roi, x_att=self.state.x_att)
        self.apply_subset_state(subset_state, override_mode=override_mode)
