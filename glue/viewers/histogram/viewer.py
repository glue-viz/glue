from glue.core.util import update_ticks
from glue.core.roi import RangeROI
from glue.core.subset import roi_to_subset_state
from glue.utils import mpl_to_datetime64

from glue.viewers.histogram.compat import update_histogram_viewer_state

__all__ = ['MatplotlibHistogramMixin']


class MatplotlibHistogramMixin(object):

    def setup_callbacks(self):
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('x_log', self._update_axes)
        self.state.add_callback('normalize', self._update_axes)
        self._update_axes()

    def _update_axes(self, *args):

        if self.state.x_att is not None:

            # Update ticks, which sets the labels to categories if components are categorical
            update_ticks(self.axes, 'x', self.state.x_kinds, self.state.x_log, self.state.x_categories)

            self.state.x_axislabel = self.state.x_att.label

        if self.state.normalize:
            self.state.y_axislabel = 'Normalized number'
        else:
            self.state.y_axislabel = 'Number'

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
