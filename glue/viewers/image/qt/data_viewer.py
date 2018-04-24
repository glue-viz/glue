from __future__ import absolute_import, division, print_function

import os

from astropy.wcs import WCS

from qtpy.QtWidgets import QMessageBox

from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import ScatterLayerStyleEditor
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.image.qt.layer_style_editor import ImageLayerStyleEditor
from glue.viewers.image.qt.layer_style_editor_subset import ImageLayerSubsetStyleEditor
from glue.viewers.image.layer_artist import ImageLayerArtist, ImageSubsetLayerArtist
from glue.viewers.image.qt.options_widget import ImageOptionsWidget
from glue.viewers.image.qt.mouse_mode import RoiClickAndDragMode
from glue.viewers.image.state import ImageViewerState
from glue.viewers.image.compat import update_image_viewer_state
from glue.utils import defer_draw

from glue.external.modest_image import imshow
from glue.viewers.image.composite_array import CompositeArray

# Import the mouse mode to make sure it gets registered
from glue.viewers.image.contrast_mouse_mode import ContrastBiasMode  # noqa
from glue.viewers.image.qt.profile_viewer_tool import ProfileViewerTool  # noqa
from glue.viewers.image.pixel_selection_mode import PixelSelectionTool  # noqa

__all__ = ['ImageViewer']


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


class ImageViewer(MatplotlibDataViewer):

    LABEL = '2D Image'
    _default_mouse_mode_cls = RoiClickAndDragMode
    _layer_style_widget_cls = {ImageLayerArtist: ImageLayerStyleEditor,
                               ImageSubsetLayerArtist: ImageLayerSubsetStyleEditor,
                               ScatterLayerArtist: ScatterLayerStyleEditor}
    _state_cls = ImageViewerState
    _options_cls = ImageOptionsWidget

    allow_duplicate_data = True

    # NOTE: _data_artist_cls and _subset_artist_cls are not defined - instead
    #       we override get_data_layer_artist and get_subset_layer_artist for
    #       more advanced logic.

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon', 'image:point_selection', 'image:contrast_bias',
             'profile-viewer']

    def __init__(self, session, parent=None, state=None):
        self._wcs_set = False
        self._changing_slice_requires_wcs_update = None
        super(ImageViewer, self).__init__(session, parent=parent, wcs=True, state=state)
        self.axes.set_adjustable('datalim')
        self.state.add_callback('x_att', self._set_wcs)
        self.state.add_callback('y_att', self._set_wcs)
        self.state.add_callback('slices', self._on_slice_change)
        self.state.add_callback('reference_data', self._set_wcs)
        self.axes._composite = CompositeArray()
        self.axes._composite_image = imshow(self.axes, self.axes._composite,
                                            origin='lower', interpolation='nearest')
        self._set_wcs()

    @defer_draw
    def update_x_ticklabel(self, *event):
        # We need to overload this here for WCSAxes
        if self._wcs_set and self.state.x_att is not None:
            axis = self.state.reference_data.ndim - self.state.x_att.axis - 1
        else:
            axis = 0
        self.axes.coords[axis].set_ticklabel(size=self.state.x_ticklabel_size)
        self.redraw()

    @defer_draw
    def update_y_ticklabel(self, *event):
        # We need to overload this here for WCSAxes
        if self._wcs_set and self.state.y_att is not None:
            axis = self.state.reference_data.ndim - self.state.y_att.axis - 1
        else:
            axis = 1
        self.axes.coords[axis].set_ticklabel(size=self.state.y_ticklabel_size)
        self.redraw()

    def closeEvent(self, *args):
        super(ImageViewer, self).closeEvent(*args)
        if self.axes._composite_image is not None:
            self.axes._composite_image.remove()
            self.axes._composite_image = None

    @defer_draw
    def _update_axes(self, *args):

        if self.state.x_att_world is not None:
            self.state.x_axislabel = self.state.x_att_world.label

        if self.state.y_att_world is not None:
            self.state.y_axislabel = self.state.y_att_world.label

        self.axes.figure.canvas.draw()

    def add_data(self, data):
        result = super(ImageViewer, self).add_data(data)
        # If this is the first layer (or the first after all layers were)
        # removed, set the WCS for the axes.
        if len(self.layers) == 1:
            self._set_wcs()
        return result

    def _on_slice_change(self, event=None):
        if self._changing_slice_requires_wcs_update:
            self._set_wcs(event=event, relim=False)

    def _set_wcs(self, event=None, relim=True):

        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return

        ref_coords = self.state.reference_data.coords

        if hasattr(ref_coords, 'wcs'):
            self.axes.reset_wcs(slices=self.state.wcsaxes_slice, wcs=ref_coords.wcs)
        elif hasattr(ref_coords, 'wcsaxes_dict'):
            self.axes.reset_wcs(slices=self.state.wcsaxes_slice, **ref_coords.wcsaxes_dict)
        else:
            self.axes.reset_wcs(slices=self.state.wcsaxes_slice,
                                wcs=get_identity_wcs(self.state.reference_data.ndim))

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
        ix = self.state.x_att.axis
        iy = self.state.y_att.axis
        x_dep = list(ref_coords.dependent_axes(ix))
        y_dep = list(ref_coords.dependent_axes(iy))
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

    def _roi_to_subset_state(self, roi):
        """ This method must be implemented in order for apply_roi from the
        parent class to work.
        """

        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return

        # TODO Does subset get applied to all data or just visible data?

        x_comp = self.state.x_att.parent.get_component(self.state.x_att)
        y_comp = self.state.y_att.parent.get_component(self.state.y_att)

        return x_comp.subset_from_roi(self.state.x_att, roi,
                                      other_comp=y_comp,
                                      other_att=self.state.y_att,
                                      coord='x')

    def _scatter_artist(self, axes, state, layer=None, layer_state=None):
        if len(self._layer_artist_container) == 0:
            QMessageBox.critical(self, "Error", "Can only add a scatter plot "
                                 "overlay once an image is present",
                                 buttons=QMessageBox.Ok)
            return None
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

        self.axes.figure.canvas.draw()

    def hide_crosshairs(self):
        if getattr(self, '_crosshairs', None) is not None:
            self._crosshairs.remove()
            self._crosshairs = None
            self.axes.figure.canvas.draw()

    def update_aspect(self, aspect=None):
        super(ImageViewer, self).update_aspect(aspect=aspect)
        if self.state.reference_data is not None and self.state.x_att is not None and self.state.y_att is not None:
            nx = self.state.reference_data.shape[self.state.x_att.axis]
            ny = self.state.reference_data.shape[self.state.y_att.axis]
            self.axes.set_xlim(-0.5, nx - 0.5)
            self.axes.set_ylim(-0.5, ny - 0.5)
            self.axes.figure.canvas.draw()

    def _script_header(self):

        imports = []
        imports.append('import matplotlib.pyplot as plt')
        imports.append('from glue.viewers.common.viz_client import init_mpl')
        imports.append('from glue.viewers.image.composite_array import CompositeArray')
        imports.append('from glue.external.modest_image import imshow')

        script = ""
        script += "fig, ax = init_mpl(wcs=True)\n"
        script += "ax.set_aspect('{0}')\n".format(self.state.aspect)

        script += '\ncomposite = CompositeArray()\n'
        script += "image = imshow(ax, composite, origin='lower', interpolation='nearest', aspect='{0}')\n\n".format(self.state.aspect)

        dindex = self.session.data_collection.index(self.state.reference_data)

        script += "ref_data = data_collection[{0}]\n".format(dindex)

        ref_coords = self.state.reference_data.coords

        if hasattr(ref_coords, 'wcs'):
            script += "ax.reset_wcs(slices={0}, wcs=ref_data.coords.wcs)\n".format(self.state.wcsaxes_slice)
        elif hasattr(ref_coords, 'wcsaxes_dict'):
            raise NotImplementedError()
        else:
            imports.append('from glue.viewers.image.qt.data_viewer import get_identity_wcs')
            script += "ax.reset_wcs(slices={0}, wcs=get_identity_wcs(ref_data.ndim))\n".format(self.state.wcsaxes_slice)

        return imports, script

    def _script_footer(self):
        imports, script = super(ImageViewer, self)._script_footer()
        options = dict(x_att_axis=0 if self.state.x_att is None else self.state.reference_data.ndim - self.state.x_att.axis - 1,
                       y_att_axis=1 if self.state.y_att is None else self.state.reference_data.ndim - self.state.y_att.axis - 1,
                       x_ticklabel_size=self.state.x_ticklabel_size,
                       y_ticklabel_size=self.state.y_ticklabel_size)
        return [], EXTRA_FOOTER.format(**options) + os.linesep * 2 + script
