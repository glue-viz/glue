import numpy as np

from astropy.visualization.wcsaxes.frame import RectangularFrame1D

from glue.core.units import UnitConverter
from glue.core.subset import roi_to_subset_state

from glue.viewers.matplotlib.viewer import SimpleMatplotlibViewer
from glue.viewers.image.viewer import get_identity_wcs
from glue.viewers.profile.state import ProfileViewerState
from glue.viewers.profile.layer_artist import ProfileLayerArtist

__all__ = ['MatplotlibProfileMixin', 'SimpleProfileViewer']


class MatplotlibProfileMixin(object):

    def setup_callbacks(self):
        # Viewers whose axes are WCSAxes get world tick labels formatted by
        # the reference data's WCS (viewers with plain axes are unaffected)
        self.state.wcsaxes = hasattr(self.axes, 'reset_wcs')
        self.state.add_callback('x_att', self._update_axes)
        self.state.add_callback('normalize', self._update_axes)
        self.state.add_callback('y_display_unit', self._update_axes)
        self.state.add_callback('x_att', self._set_wcs)
        self.state.add_callback('reference_data', self._set_wcs, echo_old=True)
        self.state.add_callback('x_display_unit', self._set_wcs)
        self.state.add_callback('slices', self._set_wcs)
        self._set_wcs()

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

    def _set_wcs(self, before=None, after=None):

        if self.state.reference_data is None or self.state.x_att_pixel is None:
            return

        # A callback event for reference_data is triggered if the choices change
        # but the actual selection doesn't - so we avoid resetting the WCS in
        # this case.
        if after is not None and before is after:
            return

        # x limits saved in the other mode (world/display values vs pixel
        # coordinates) cannot be reinterpreted - reset them. This matters for
        # sessions restored into a viewer whose mode differs from the one
        # they were saved in.
        if self.state.x_limits_pixel != self.state.wcsaxes_active:
            self.state._reset_x_limits()

        if not self.state.wcsaxes:
            return

        if self.state.wcsaxes_active:
            wcs = self.state.reference_data.coords
            slices = self.state.wcsaxes_slice
        else:
            # A plain numeric axis - the profile x values are plotted directly
            wcs = get_identity_wcs(1)
            slices = ('x',)

        # Avoid needless WCS resets, e.g. for unit changes within a mode
        wcs_key = (self.state.wcsaxes_active,
                   id(wcs) if self.state.wcsaxes_active else None, slices)
        if getattr(self, '_last_wcs_key', None) == wcs_key:
            return
        self._last_wcs_key = wcs_key

        self.axes.frame_class = RectangularFrame1D
        self.axes.reset_wcs(slices=slices, wcs=wcs)

        # The y axis is a plain matplotlib axis, which WCSAxes hid when the
        # axes were created with the default 2D frame - unhide it
        self.axes.yaxis.set_visible(True)

        # Reset the axis labels to match the fact that the new axes have no
        # labels, so that _update_axes re-applies them
        self.state.x_axislabel = ''
        self.state.y_axislabel = ''

        self._update_appearance_from_settings()
        self._update_axes()

        self.update_x_ticklabel()
        self.update_y_ticklabel()

    def update_x_ticklabel(self, *event):
        if not self.state.wcsaxes:
            return super().update_x_ticklabel(*event)
        # tick_params silently does nothing on a 1D-frame WCSAxes, so use the
        # coordinate helper of the world axis correlated with the profile
        # dimension (ax.coords is in WCS axis order)
        if self.state.wcsaxes_active and self.state.x_att_pixel is not None:
            axis = self.state.reference_data.ndim - self.state.x_att_pixel.axis - 1
        else:
            axis = 0
        self.axes.coords[axis].set_ticklabel(size=self.state.x_ticklabel_size)
        self.redraw()

    def update_y_ticklabel(self, *event):
        if not self.state.wcsaxes:
            return super().update_y_ticklabel(*event)
        self.axes.yaxis.set_tick_params(labelsize=self.state.y_ticklabel_size)
        self.axes.yaxis.get_offset_text().set_fontsize(self.state.y_ticklabel_size)
        self.redraw()

    def apply_roi(self, roi, override_mode=None):

        # Force redraw to get rid of ROI. We do this because applying the
        # subset state below might end up not having an effect on the viewer,
        # for example there may not be any layers, or the active subset may not
        # be one of the layers. So we just explicitly redraw here to make sure
        # a redraw will happen after this method is called.
        self.redraw()

        if len(self.layers) == 0:
            return

        if self.state.wcsaxes_active:
            # The profile is plotted in pixel coordinates
            subset_state = roi_to_subset_state(roi, x_att=self.state.x_att_pixel)
        else:
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

    def _script_header(self):

        if not self.state.wcsaxes:
            return super()._script_header()

        imports = ['import matplotlib.pyplot as plt',
                   'from glue.viewers.matplotlib.mpl_axes import init_mpl',
                   'from glue.viewers.matplotlib.mpl_axes import set_figure_colors',
                   'from astropy.visualization.wcsaxes.frame import RectangularFrame1D']

        script = ""
        script += "fig, ax = init_mpl(wcs=True)\n"
        script += "ax.frame_class = RectangularFrame1D\n"

        if self.state.wcsaxes_active:
            dindex = self.session.data_collection.index(self.state.reference_data)
            script += f"ref_data = data_collection[{dindex}]\n"
            ref_wcs = "ref_data.coords"
            slices = self.state.wcsaxes_slice
        else:
            imports.append('from glue.viewers.image.viewer import get_identity_wcs')
            ref_wcs = "get_identity_wcs(1)"
            slices = ('x',)

        script += f"ax.reset_wcs(slices={slices}, wcs={ref_wcs})\n"
        script += "ax.yaxis.set_visible(True)\n\n"

        script += "# for the legend\n"
        script += "legend_handles = []\n"
        script += "legend_labels = []\n"
        script += "legend_handler_dict = dict()\n\n"

        return imports, script

    def _script_footer(self):

        imports, script = super()._script_footer()

        if not self.state.wcsaxes:
            return imports, script

        # tick_params in the generic footer does nothing on a 1D-frame WCSAxes
        if self.state.wcsaxes_active and self.state.x_att_pixel is not None:
            axis = self.state.reference_data.ndim - self.state.x_att_pixel.axis - 1
        else:
            axis = 0
        extra = (f"ax.coords[{axis}].set_ticklabel(size={self.state.x_ticklabel_size})\n"
                 f"ax.yaxis.set_tick_params(labelsize={self.state.y_ticklabel_size})\n\n")

        return imports, extra + script


class SimpleProfileViewer(MatplotlibProfileMixin, SimpleMatplotlibViewer):

    _state_cls = ProfileViewerState
    _data_artist_cls = ProfileLayerArtist
    _subset_artist_cls = ProfileLayerArtist

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        MatplotlibProfileMixin.setup_callbacks(self)
