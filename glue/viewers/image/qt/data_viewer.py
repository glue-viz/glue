from __future__ import absolute_import, division, print_function

from astropy.wcs import WCS

from qtpy.QtWidgets import QMessageBox

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.utils import defer_draw

from glue.core.coordinates import WCSCoordinates
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import ScatterLayerStyleEditor
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.image.qt.layer_style_editor import ImageLayerStyleEditor
from glue.viewers.image.qt.layer_style_editor_subset import ImageLayerSubsetStyleEditor
from glue.viewers.image.layer_artist import ImageLayerArtist, ImageSubsetLayerArtist
from glue.viewers.image.qt.options_widget import ImageOptionsWidget
from glue.viewers.image.state import ImageViewerState
from glue.viewers.image.compat import update_image_viewer_state

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
                               ScatterLayerArtist: ScatterLayerStyleEditor}
    _state_cls = ImageViewerState
    _options_cls = ImageOptionsWidget

    allow_duplicate_data = True

    # NOTE: _data_artist_cls and _subset_artist_cls are implemented as methods

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon', 'image:contrast_bias']

    def __init__(self, session, parent=None, state=None):
        super(ImageViewer, self).__init__(session, parent=parent, wcs=True, state=state)
        self.axes.set_adjustable('datalim')
        self.state.add_callback('aspect', self._set_aspect)
        self.state.add_callback('x_att', self._set_wcs)
        self.state.add_callback('y_att', self._set_wcs)
        self.state.add_callback('slices', self._set_wcs)
        self.state.add_callback('reference_data', self._set_wcs)
        self.state.add_callback('reference_data', self._set_aspect)
        self.axes._composite = CompositeArray()
        self.axes._composite_image = imshow(self.axes, self.axes._composite,
                                            origin='lower', interpolation='nearest')

    @defer_draw
    def _update_axes(self, *args):

        if self.state.x_att_world is not None:
            self.axes.set_xlabel(self.state.x_att_world.label)

        if self.state.y_att_world is not None:
            self.axes.set_ylabel(self.state.y_att_world.label)

        self.axes.figure.canvas.draw()

    @defer_draw
    def add_data(self, data):
        result = super(ImageViewer, self).add_data(data)
        # If this is the first layer (or the first after all layers were)
        # removed, set the WCS for the axes and the aspect (which also sets)
        # the limits.
        if len(self.layers) == 1:
            self._set_wcs()
            self._set_aspect()
        return result

    @defer_draw
    def _set_aspect(self, *args):
        self.axes.set_aspect(self.state.aspect)
        if self.axes._composite.shape is not None:
            ny, nx = self.axes._composite.shape
            self.axes.set_xlim(-0.5, nx - 0.5)
            self.axes.set_ylim(-0.5, ny - 0.5)
            # FIXME: for a reason I don't quite understand, dataLim doesn't
            # get updated immediately here, which means that there are then
            # issues in the first draw of the image (the limits are such that
            # only part of the image is shown). We just set dataLim manually
            # to avoid this issue.
            self.axes.dataLim.intervalx = self.axes.get_xlim()
            self.axes.dataLim.intervaly = self.axes.get_ylim()
        self.axes.figure.canvas.draw()

    @defer_draw
    def _set_wcs(self, *args):
        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return
        ref_coords = self.state.reference_data.coords
        if isinstance(ref_coords, WCSCoordinates):
            self.axes.reset_wcs(ref_coords.wcs, slices=self.state.wcsaxes_slice)
        else:
            self.axes.reset_wcs(IDENTITY_WCS)
        self._update_appearance_from_settings()
        self._update_axes()

    def apply_roi(self, roi):

        # TODO: move this to state class?

        # TODO: add back command stack here so as to be able to undo?
        # cmd = command.ApplyROI(client=self.client, roi=roi)
        # self._session.command_stack.do(cmd)

        # TODO Does subset get applied to all data or just visible data?

        x_comp = self.state.x_att.parent.get_component(self.state.x_att)
        y_comp = self.state.y_att.parent.get_component(self.state.y_att)

        subset_state = x_comp.subset_from_roi(self.state.x_att, roi,
                                              other_comp=y_comp,
                                              other_att=self.state.y_att,
                                              coord='x')

        mode = EditSubsetMode()
        mode.update(self._data, subset_state)

    def _scatter_artist(self, axes, state, layer=None):
        if len(self._layer_artist_container) == 0:
            QMessageBox.critical(self, "Error", "Can only add a scatter plot "
                                 "overlay once an image is present",
                                 buttons=QMessageBox.Ok)
            return None
        return ScatterLayerArtist(axes, state, layer=layer)

    def _data_artist_cls(self, axes, state, layer=None):
        if layer.ndim == 1:
            return self._scatter_artist(axes, state, layer=layer)
        else:
            return ImageLayerArtist(axes, state, layer=layer)

    def _subset_artist_cls(self, axes, state, layer=None):
        if layer.ndim == 1:
            return self._scatter_artist(axes, state, layer=layer)
        else:
            return ImageSubsetLayerArtist(axes, state, layer=layer)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_image_viewer_state(rec, context)

    @defer_draw
    def show_crosshairs(self, x, y):

        if getattr(self, '_crosshairs', None) is not None:
            self._crosshairs.remove()

        self._crosshairs, = self.axes.plot([x], [y], '+', ms=12,
                                           mfc='none', mec='#d32d26',
                                           mew=1, zorder=100)

        self.axes.figure.canvas.draw()
