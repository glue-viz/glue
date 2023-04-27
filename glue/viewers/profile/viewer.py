import numpy as np

from glue.core.units import UnitConverter
from glue.core.subset import roi_to_subset_state

__all__ = ['MatplotlibProfileMixin']


class MatplotlibProfileMixin(object):

    def setup_callbacks(self):
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('normalize', self._update_axes)
        self.state.add_callback('y_display_unit', self._update_axes)

    def _update_axes(self, *args):

        if self.state.x_att is not None:
            self.state.x_axislabel = self.state.x_att.label

        if self.state.normalize:
            self.state.y_axislabel = 'Normalized data values'
        else:
            if self.state.y_display_unit:
                self.state.y_axislabel = f'Data values [{self.state.y_display_unit}]'
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

        # Apply inverse unit conversion, converting from display to native units
        converter = UnitConverter()
        cmin, cmax = converter.to_native(self.state.reference_data,
                                         self.state.x_att, np.array([roi.min, roi.max]),
                                         self.state.x_display_unit)

        # Sometimes unit conversions can cause the min/max to be swapped
        if cmin > cmax:
            cmin, cmax = cmax, cmin

        roi.min = cmin
        roi.max = cmax

        subset_state = roi_to_subset_state(roi, x_att=self.state.x_att)
        self.apply_subset_state(subset_state, override_mode=override_mode)
