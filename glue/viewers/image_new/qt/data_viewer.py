from __future__ import absolute_import, division, print_function

from astropy.wcs import WCS

from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core import Data
from glue.utils import defer_draw

from glue.core.coordinates import WCSCoordinates
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.image_new.qt.layer_style_editor import ImageLayerStyleEditor
from glue.viewers.image_new.qt.layer_style_editor_subset import ImageLayerSubsetStyleEditor
from glue.viewers.image_new.layer_artist import ImageLayerArtist, ImageSubsetLayerArtist
from glue.viewers.image_new.qt.options_widget import ImageOptionsWidget
from glue.viewers.image_new.state import ImageViewerState

from glue.external.modest_image import imshow
from glue.viewers.image_new.composite_array import CompositeArray

# Import the mouse mode to make sure it gets registered
from glue.viewers.image_new.contrast_mouse_mode import ContrastBiasMode  # noqa

__all__ = ['ImageViewer']


class ImageViewer(MatplotlibDataViewer):

    LABEL = '2D Image'
    _toolbar_cls = MatplotlibViewerToolbar
    _layer_style_widget_cls = {ImageLayerArtist: ImageLayerStyleEditor,
                               ImageSubsetLayerArtist: ImageLayerSubsetStyleEditor}
    _state_cls = ImageViewerState
    _options_cls = ImageOptionsWidget
    _data_artist_cls = ImageLayerArtist
    _subset_artist_cls = ImageSubsetLayerArtist

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon', 'image:contrast_bias']

    def __init__(self, session, parent=None):
        super(ImageViewer, self).__init__(session, parent=parent, wcs=True)
        self.state.add_callback('aspect', self._set_aspect)
        self.state.add_callback('x_att', self._set_wcs)
        self.state.add_callback('y_att', self._set_wcs)
        self.state.add_callback('slices', self._set_wcs)
        self.state.add_callback('reference_data', self._set_wcs)
        self.axes._composite = CompositeArray(self.axes)
        self.axes._composite_image = imshow(self.axes, self.axes._composite,
                                            origin='lower', interpolation='nearest')

    @defer_draw
    def _update_axes(self, *args):

        if self.state.x_att_world is not None:
            self.axes.set_xlabel(self.state.x_att_world.label)

        if self.state.y_att_world is not None:
            self.axes.set_ylabel(self.state.y_att_world.label)

        self.axes.figure.canvas.draw()

    def _set_aspect(self, *args):
        self.axes.set_aspect(self.state.aspect)
        self.axes.figure.canvas.draw()

    def _set_wcs(self, *args):
        if self.state.x_att is None or self.state.y_att is None or self.state.reference_data is None:
            return
        ref_coords = self.state.reference_data.coords
        if isinstance(ref_coords, WCSCoordinates):
            self.axes.reset_wcs(ref_coords.wcs, slices=self.state.wcsaxes_slice)
        self._update_axes()

    def apply_roi(self, roi):

        # TODO: move this to state class?

        # TODO: add back command stack here so as to be able to undo?
        # cmd = command.ApplyROI(client=self.client, roi=roi)
        # self._session.command_stack.do(cmd)

        # TODO Does subset get applied to all data or just visible data?

        for layer_artist in self._layer_artist_container:

            if not isinstance(layer_artist.layer, Data):
                continue

            x_comp = layer_artist.layer.get_component(self.state.x_att)
            y_comp = layer_artist.layer.get_component(self.state.y_att)

            subset_state = x_comp.subset_from_roi(self.state.x_att, roi,
                                                  other_comp=y_comp,
                                                  other_att=self.state.y_att,
                                                  coord='x')

            mode = EditSubsetMode()
            mode.update(self._data, subset_state, focus_data=layer_artist.layer)
