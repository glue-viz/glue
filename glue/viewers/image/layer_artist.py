from __future__ import absolute_import, division, print_function

import uuid
import numpy as np

from glue.utils import defer_draw

from glue.viewers.image.state import ImageLayerState, ImageSubsetLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute
from glue.utils import color2rgb
from glue.core.link_manager import is_equivalent_cid
from glue.core import Data, HubListener
from glue.core.message import ComponentsChangedMessage


class BaseImageLayerArtist(MatplotlibLayerArtist, HubListener):

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(BaseImageLayerArtist, self).__init__(axes, viewer_state,
                                                   layer_state=layer_state, layer=layer)

        self.reset_cache()

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_image)
        self.state.add_global_callback(self._update_image)

        # TODO: following is temporary
        self.state.data_collection = self._viewer_state.data_collection
        self.data_collection = self._viewer_state.data_collection

        def is_data_object(message):
            if isinstance(self.layer, Data):
                return message.sender is self.layer
            else:
                return message.sender is self.layer.data

        self.data_collection.hub.subscribe(self, ComponentsChangedMessage,
                                           handler=self._update_compatibility,
                                           filter=is_data_object)

        self._update_compatibility()

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    def _update_image(self, force=False, **kwargs):
        raise NotImplementedError()

    @defer_draw
    def _update_compatibility(self, *args, **kwargs):
        """
        Determine compatibility of data with reference data. For the data to be
        compatible with the reference data, the number of dimensions has to
        match and the pixel component IDs have to be equivalent.
        """

        if self.layer is self._viewer_state.reference_data:
            self._compatible_with_reference_data = True
            self.enable()
            return

        # Check whether the pixel component IDs of the dataset are equivalent
        # to that of the reference dataset. In future this is where we could
        # allow for these to be different and implement reprojection.
        if self.layer.ndim != self._viewer_state.reference_data.ndim:
            self._compatible_with_reference_data = False
            self.disable('Data dimensions do not match reference data')
            return

        # Determine whether pixel component IDs are equivalent

        pids = self.layer.pixel_component_ids
        pids_ref = self._viewer_state.reference_data.pixel_component_ids

        if isinstance(self.layer, Data):
            data = self.layer
        else:
            data = self.layer.data

        for i in range(data.ndim):
            if not is_equivalent_cid(data, pids[i], pids_ref[i]):
                self._compatible_with_reference_data = False
                self.disable('Pixel component IDs do not match. You can try '
                             'fixing this by linking the pixel component IDs '
                             'of this dataset with those of the reference '
                             'dataset.')
                return

        self._compatible_with_reference_data = True
        self.enable()


class ImageLayerArtist(BaseImageLayerArtist):

    _layer_state_cls = ImageLayerState

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ImageLayerArtist, self).__init__(axes, viewer_state,
                                               layer_state=layer_state, layer=layer)

        # We use a custom object to deal with the compositing of images, and we
        # store it as a private attribute of the axes to make sure it is
        # accessible for all layer artists.
        self.uuid = str(uuid.uuid4())
        self.composite = self.axes._composite
        self.composite.allocate(self.uuid)
        self.composite.set(self.uuid, array=self.get_image_data)
        self.composite_image = self.axes._composite_image

    def get_image_data(self):

        if not self._compatible_with_reference_data:
            return None

        try:
            # FIXME: is the following slow? Should slide at same time?
            image = self.layer[self.state.attribute]
        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self.state.attribute)
            return None
        else:
            self._enabled = True

        slices, transpose = self._viewer_state.numpy_slice_and_transpose
        image = image[slices]
        if transpose:
            image = image.transpose()

        return image

    def _update_image_data(self):
        self.composite_image.invalidate_cache()
        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        if self._viewer_state.color_mode == 'Colormaps':
            color = self.state.cmap
        else:
            color = self.state.color

        self.composite.set(self.uuid,
                           clim=(self.state.v_min, self.state.v_max),
                           visible=self.state.visible,
                           zorder=self.state.zorder,
                           color=color,
                           contrast=self.state.contrast,
                           bias=self.state.bias,
                           alpha=self.state.alpha,
                           stretch=self.state.stretch)

        self.composite_image.invalidate_cache()

        self.redraw()

    def _update_image(self, force=False, **kwargs):

        if self.state.attribute is None or self.state.layer is None:
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to x_att changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_histogram is really only called once
        # then we could consider simplifying this. Until then, we manually keep track
        # of which properties have changed.

        changed = set()

        if not force:

            for key, value in self._viewer_state.as_dict().items():
                if value != self._last_viewer_state.get(key, None):
                    changed.add(key)

            for key, value in self.state.as_dict().items():
                if value != self._last_layer_state.get(key, None):
                    changed.add(key)

        self._last_viewer_state.update(self._viewer_state.as_dict())
        self._last_layer_state.update(self.state.as_dict())

        if 'reference_data' in changed or 'layer' in changed:
            self._update_compatibility()

        if force or any(prop in changed for prop in ('layer', 'attribute',
                                                     'slices', 'x_att', 'y_att')):
            self._update_image_data()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('v_min', 'v_max', 'contrast',
                                                     'bias', 'alpha', 'color_mode',
                                                     'cmap', 'color', 'zorder',
                                                     'visible', 'stretch')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):

        self._update_image(force=True)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()


class ImageSubsetLayerArtist(BaseImageLayerArtist):

    _layer_state_cls = ImageSubsetLayerState

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ImageSubsetLayerArtist, self).__init__(axes, viewer_state,
                                                     layer_state=layer_state, layer=layer)

        self.mpl_image = self.axes.imshow([[0.]],
                                          origin='lower', interpolation='nearest',
                                          vmin=0, vmax=1, aspect=self._viewer_state.aspect)

    def _get_image_data(self):

        view, transpose = self._viewer_state.numpy_slice_and_transpose

        mask = self.layer.to_mask(view=view)

        if transpose:
            mask = mask.transpose()

        r, g, b = color2rgb(self.state.color)
        mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = (255 * mask).astype(np.uint8)

        return mask

    def _update_image_data(self):

        if self._compatible_with_reference_data:

            try:
                data = self._get_image_data()
            except IncompatibleAttribute:
                self.disable_invalid_attributes(self.state.attribute)
                data = np.array([[np.nan]])
            else:
                self._enabled = True

        else:
            data = np.array([[np.nan]])

        self.mpl_image.set_data(data)
        self.mpl_image.set_extent([-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5])
        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        # TODO: deal with color using a colormap instead of having to change data

        self.mpl_image.set_visible(self.state.visible)
        self.mpl_image.set_zorder(self.state.zorder)
        self.mpl_image.set_alpha(self.state.alpha)

        self.redraw()

    def _update_image(self, force=False, **kwargs):

        if self.state.layer is None:
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to x_att changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_histogram is really only called once
        # then we could consider simplifying this. Until then, we manually keep track
        # of which properties have changed.

        changed = set()

        if not force:

            for key, value in self._viewer_state.as_dict().items():
                if value != self._last_viewer_state.get(key, None):
                    changed.add(key)

            for key, value in self.state.as_dict().items():
                if value != self._last_layer_state.get(key, None):
                    changed.add(key)

        self._last_viewer_state.update(self._viewer_state.as_dict())
        self._last_layer_state.update(self.state.as_dict())

        if 'reference_data' in changed or 'layer' in changed:
            self._update_compatibility()

        if force or any(prop in changed for prop in ('layer', 'attribute', 'color',
                                                     'x_att', 'y_att', 'slices')):
            self._update_image_data()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('zorder', 'visible', 'alpha')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):

        # TODO: determine why this gets called when changing the transparency slider

        self._update_image(force=True)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
