from __future__ import absolute_import, division, print_function

from astropy.wcs import WCS

from qtpy.QtWidgets import QMessageBox

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode

from glue.core import command
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import GenericLayerStyleEditor
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.image.qt.layer_style_editor import ImageLayerStyleEditor
from glue.viewers.image.qt.layer_style_editor_subset import ImageLayerSubsetStyleEditor
from glue.viewers.image.layer_artist import ImageLayerArtist, ImageSubsetLayerArtist
from glue.viewers.image.qt.options_widget import ImageOptionsWidget
from glue.viewers.image.state import ImageViewerState
from glue.viewers.image.compat import update_image_viewer_state
from glue.external.echo import delay_callback

from glue.external.modest_image import imshow
from glue.viewers.image.composite_array import CompositeArray

# Import the mouse mode to make sure it gets registered
from glue.viewers.image.contrast_mouse_mode import ContrastBiasMode  # noqa

__all__ = ['ImageViewer']

IDENTITY_WCS = WCS(naxis=2)
IDENTITY_WCS.wcs.ctype = ["X", "Y"]
IDENTITY_WCS.wcs.crval = [0., 0.]
IDENTITY_WCS.wcs.crpix = [1., 1.]
IDENTITY_WCS.wcs.cdelt = [1., 1.]


class ImageViewer(MatplotlibDataViewer):

    LABEL = '2D Image'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = {ImageLayerArtist: ImageLayerStyleEditor,
                               ImageSubsetLayerArtist: ImageLayerSubsetStyleEditor,
                               ScatterLayerArtist: GenericLayerStyleEditor}
    _state_cls = ImageViewerState
    _options_cls = ImageOptionsWidget

    allow_duplicate_data = True

    # NOTE: _data_artist_cls and _subset_artist_cls are not defined - instead
    #       we override get_data_layer_artist and get_subset_layer_artist for
    #       more advanced logic.

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon', 'image:contrast_bias']

    def __init__(self, session, parent=None, state=None):
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

    def close(self, **kwargs):
        super(ImageViewer, self).close(**kwargs)
        if self.axes._composite_image is not None:
            self.axes._composite_image.remove()
            self.axes._composite_image = None

    def _update_axes(self, *args):

        if self.state.x_att_world is not None:
            self.axes.set_xlabel(self.state.x_att_world.label)

        if self.state.y_att_world is not None:
            self.axes.set_ylabel(self.state.y_att_world.label)

        self.axes.figure.canvas.draw()

    def add_data(self, data):
        result = super(ImageViewer, self).add_data(data)
        # If this is the first layer (or the first after all layers were)
        # removed, set the WCS for the axes.
        if len(self.layers) == 1:
            self._set_wcs()
        return result

    def _on_slice_change(self, event=None):

        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return

        coords = self.state.reference_data.coords
        ix = self.state.x_att.axis
        iy = self.state.y_att.axis
        x_dep = list(coords.dependent_axes(ix))
        y_dep = list(coords.dependent_axes(iy))
        if ix in x_dep:
            x_dep.remove(ix)
        if iy in x_dep:
            x_dep.remove(iy)
        if ix in y_dep:
            y_dep.remove(ix)
        if iy in y_dep:
            y_dep.remove(iy)
        if x_dep or y_dep:
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
            self.axes.reset_wcs(IDENTITY_WCS)

        self._update_appearance_from_settings()
        self._update_axes()

        if relim:

            nx = self.state.reference_data.shape[self.state.x_att.axis]
            ny = self.state.reference_data.shape[self.state.y_att.axis]

            with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
                self.state.x_min = -0.5
                self.state.x_max = nx - 0.5
                self.state.y_min = -0.5
                self.state.y_max = ny - 0.5

    # TODO: move some of the ROI stuff to state class?

    def apply_roi(self, roi):
        cmd = command.ApplyROI(data_collection=self._data,
                               roi=roi, apply_func=self._apply_roi)
        self._session.command_stack.do(cmd)

    def _apply_roi(self, roi):

        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return

        # TODO Does subset get applied to all data or just visible data?

        x_comp = self.state.x_att.parent.get_component(self.state.x_att)
        y_comp = self.state.y_att.parent.get_component(self.state.y_att)

        subset_state = x_comp.subset_from_roi(self.state.x_att, roi,
                                              other_comp=y_comp,
                                              other_att=self.state.y_att,
                                              coord='x')

        mode = EditSubsetMode()
        mode.update(self._data, subset_state)

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
