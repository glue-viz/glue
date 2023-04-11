import os

from astropy.wcs import WCS

from glue.core.subset import roi_to_subset_state
from glue.core.coordinates import Coordinates, LegacyCoordinates
from glue.core.coordinate_helpers import dependent_axes

from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.image.layer_artist import ImageLayerArtist, ImageSubsetLayerArtist
from glue.viewers.image.compat import update_image_viewer_state

from glue.viewers.image.frb_artist import imshow
from glue.viewers.image.composite_array import CompositeArray

__all__ = ['MatplotlibImageMixin']


def get_identity_wcs(naxis):

    wcs = WCS(naxis=naxis)
    wcs.wcs.ctype = ['X'] * naxis
    wcs.wcs.crval = [0.] * naxis
    wcs.wcs.crpix = [1.] * naxis
    wcs.wcs.cdelt = [1.] * naxis

    return wcs


EXTRA_FOOTER = """
# Set tick label size - for now tick_params (called lower down) doesn't work
# properly, but these lines won't be needed in future.
ax.coords[{x_att_axis}].set_ticklabel(size={x_ticklabel_size})
ax.coords[{y_att_axis}].set_ticklabel(size={y_ticklabel_size})
""".strip()


class MatplotlibImageMixin(object):

    def setup_callbacks(self):
        self._wcs_set = False
        self._changing_slice_requires_wcs_update = None
        self.axes.set_adjustable('datalim')
        self.state.add_callback('x_att', self._set_wcs)
        self.state.add_callback('y_att', self._set_wcs)
        self.state.add_callback('slices', self._on_slice_change)
        self.state.add_callback('reference_data', self._set_wcs, echo_old=True)
        self.axes._composite = CompositeArray()
        self.axes._composite_image = imshow(self.axes, self.axes._composite, aspect='auto',
                                            origin='lower', interpolation='nearest')
        self._set_wcs()

    def update_x_ticklabel(self, *event):
        # We need to overload this here for WCSAxes
        if hasattr(self, '_wcs_set') and self._wcs_set and self.state.x_att is not None:
            axis = self.state.reference_data.ndim - self.state.x_att.axis - 1
        else:
            axis = 0
        self.axes.coords[axis].set_ticklabel(size=self.state.x_ticklabel_size)
        self.redraw()

    def update_y_ticklabel(self, *event):
        # We need to overload this here for WCSAxes
        if hasattr(self, '_wcs_set') and self._wcs_set and self.state.y_att is not None:
            axis = self.state.reference_data.ndim - self.state.y_att.axis - 1
        else:
            axis = 1
        self.axes.coords[axis].set_ticklabel(size=self.state.y_ticklabel_size)
        self.redraw()

    def _update_axes(self, *args):

        if self.state.x_att_world is not None:
            self.state.x_axislabel = self.state.x_att_world.label

        if self.state.y_att_world is not None:
            self.state.y_axislabel = self.state.y_att_world.label

        self.axes.figure.canvas.draw_idle()

    def add_data(self, data):
        result = super(MatplotlibImageMixin, self).add_data(data)
        # If this is the first layer (or the first after all layers were)
        # removed, set the WCS for the axes.
        if len(self.layers) == 1:
            self._set_wcs()
        return result

    def _update_data_numerical(self, *args, **kwargs):
        super()._update_data_numerical(*args, **kwargs)
        self.state._reference_data_changed(force=True)

    def _on_slice_change(self, event=None):
        if self._changing_slice_requires_wcs_update:
            self._set_wcs(relim=False)

    def _set_wcs(self, before=None, after=None, relim=True):

        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return

        # A callback event for reference_data is triggered if the choices change
        # but the actual selection doesn't - so we avoid resetting the WCS in
        # this case.
        if after is not None and before is after:
            return

        ref_coords = getattr(self.state.reference_data, 'coords', None)

        if ref_coords is None or isinstance(ref_coords, LegacyCoordinates):
            self.axes.reset_wcs(slices=self.state.wcsaxes_slice,
                                wcs=get_identity_wcs(self.state.reference_data.ndim))
        else:
            self.axes.reset_wcs(slices=self.state.wcsaxes_slice, wcs=ref_coords)

        # Reset the axis labels to match the fact that the new axes have no labels
        self.state.x_axislabel = ''
        self.state.y_axislabel = ''

        self._update_appearance_from_settings()
        self._update_axes()

        self.update_x_ticklabel()
        self.update_y_ticklabel()

        if relim:
            self.state.reset_limits()

        # Determine whether changing slices requires changing the WCS
        if ref_coords is None or type(ref_coords) == Coordinates:
            self._changing_slice_requires_wcs_update = False
        else:
            ix = self.state.x_att.axis
            iy = self.state.y_att.axis
            x_dep = list(dependent_axes(ref_coords, ix))
            y_dep = list(dependent_axes(ref_coords, iy))
            if ix in x_dep:
                x_dep.remove(ix)
            if iy in x_dep:
                x_dep.remove(iy)
            if ix in y_dep:
                y_dep.remove(ix)
            if iy in y_dep:
                y_dep.remove(iy)
            self._changing_slice_requires_wcs_update = bool(x_dep or y_dep)

        self._wcs_set = True

    def apply_roi(self, roi, override_mode=None):

        # Force redraw to get rid of ROI. We do this because applying the
        # subset state below might end up not having an effect on the viewer,
        # for example there may not be any layers, or the active subset may not
        # be one of the layers. So we just explicitly redraw here to make sure
        # a redraw will happen after this method is called.
        self.redraw()

        if len(self.layers) == 0:
            return

        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return

        subset_state = roi_to_subset_state(roi,
                                           x_att=self.state.x_att,
                                           y_att=self.state.y_att)

        self.apply_subset_state(subset_state, override_mode=override_mode)

    def _scatter_artist(self, axes, state, layer=None, layer_state=None):
        if len(self._layer_artist_container) == 0:
            raise Exception("Can only add a scatter plot overlay once an image is present")
        return ScatterLayerArtist(axes, state, layer=layer, layer_state=None)

    def get_data_layer_artist(self, layer=None, layer_state=None):
        if layer.ndim == 1:
            cls = self._scatter_artist
        else:
            cls = ImageLayerArtist
        return self.get_layer_artist(cls, layer=layer, layer_state=layer_state)

    def get_subset_layer_artist(self, layer=None, layer_state=None):
        if layer.ndim == 1:
            cls = self._scatter_artist
        else:
            cls = ImageSubsetLayerArtist
        return self.get_layer_artist(cls, layer=layer, layer_state=layer_state)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_image_viewer_state(rec, context)

    def show_crosshairs(self, x, y):

        if getattr(self, '_crosshairs', None) is not None:
            self._crosshairs.remove()

        self._crosshairs, = self.axes.plot([x], [y], '+', ms=12,
                                           mfc='none', mec='#d32d26',
                                           mew=1, zorder=100)

        self.axes.figure.canvas.draw_idle()

    def hide_crosshairs(self):
        if getattr(self, '_crosshairs', None) is not None:
            self._crosshairs.remove()
            self._crosshairs = None
            self.axes.figure.canvas.draw_idle()

    def _script_header(self):

        imports = []
        imports.append('import matplotlib.pyplot as plt')
        imports.append('from glue.viewers.matplotlib.mpl_axes import init_mpl')
        imports.append('from glue.viewers.image.composite_array import CompositeArray')
        imports.append('from glue.viewers.image.frb_artist import imshow')
        imports.append('from glue.viewers.matplotlib.mpl_axes import set_figure_colors')

        script = ""
        script += "fig, ax = init_mpl(wcs=True)\n"
        script += f"ax.set_aspect('{self.state.aspect}')\n"

        script += '\ncomposite = CompositeArray()\n'
        script += f"image = imshow(ax, composite, origin='lower', interpolation='nearest', aspect='{self.state.aspect}')\n\n"

        dindex = self.session.data_collection.index(self.state.reference_data)

        script += f"ref_data = data_collection[{dindex}]\n"

        if isinstance(self.state.reference_data.coords, (LegacyCoordinates, type(None))):
            imports.append('from glue.viewers.image.viewer import get_identity_wcs')
            ref_wcs = "get_identity_wcs(ref_data.ndim)"
        else:
            ref_wcs = "ref_data.coords"
        script += f"ax.reset_wcs(slices={self.state.wcsaxes_slice}, wcs={ref_wcs})\n"

        script += "# for the legend\n"
        script += "legend_handles = []\n"
        script += "legend_labels = []\n"
        script += "legend_handler_dict = dict()\n\n"

        return imports, script

    def _script_footer(self):
        imports, script = super(MatplotlibImageMixin, self)._script_footer()
        options = dict(x_att_axis=0 if self.state.x_att is None else self.state.reference_data.ndim - self.state.x_att.axis - 1,
                       y_att_axis=1 if self.state.y_att is None else self.state.reference_data.ndim - self.state.y_att.axis - 1,
                       x_ticklabel_size=self.state.x_ticklabel_size,
                       y_ticklabel_size=self.state.y_ticklabel_size)
        return [], EXTRA_FOOTER.format(**options) + os.linesep * 2 + script
