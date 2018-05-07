from __future__ import absolute_import, division, print_function

import uuid
import weakref

import numpy as np

from glue.utils import defer_draw, broadcast_to

from glue.viewers.image.state import ImageLayerState, ImageSubsetLayerState
from glue.viewers.image.python_export import python_export_image_layer, python_export_image_subset_layer
from glue.viewers.image.pixel_selection_mode import PixelSubsetState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute
from glue.utils import color2rgb
from glue.core import Data, HubListener
from glue.core.message import (ComponentsChangedMessage,
                               ExternallyDerivableComponentsChangedMessage,
                               PixelAlignedDataChangedMessage)
from glue.external.modest_image import imshow


class BaseImageLayerArtist(MatplotlibLayerArtist, HubListener):

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(BaseImageLayerArtist, self).__init__(axes, viewer_state,
                                                   layer_state=layer_state, layer=layer)

        self.reset_cache()

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_image)
        self.state.add_global_callback(self._update_image)

        self.layer.hub.subscribe(self, ComponentsChangedMessage,
                                 handler=self.update,
                                 filter=self._is_data_object)

        self.layer.hub.subscribe(self, ExternallyDerivableComponentsChangedMessage,
                                 handler=self.update,
                                 filter=self._is_data_object)

        self.layer.hub.subscribe(self, PixelAlignedDataChangedMessage,
                                 handler=self.update,
                                 filter=self._is_data_object)

    def _is_data_object(self, message):
        if isinstance(self.layer, Data):
            return message.sender is self.layer
        else:
            return message.sender is self.layer.data

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    def _update_image(self, force=False, **kwargs):
        raise NotImplementedError()


class ImageLayerArtist(BaseImageLayerArtist):

    _layer_state_cls = ImageLayerState
    _python_exporter = python_export_image_layer

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ImageLayerArtist, self).__init__(axes, viewer_state,
                                               layer_state=layer_state, layer=layer)

        # We use a custom object to deal with the compositing of images, and we
        # store it as a private attribute of the axes to make sure it is
        # accessible for all layer artists.
        self.uuid = str(uuid.uuid4())
        self.composite = self.axes._composite
        self.composite.allocate(self.uuid)
        self.composite.set(self.uuid, array=self.get_image_data,
                           shape=self.get_image_shape)
        self.composite_image = self.axes._composite_image

    @property
    def label(self):
        return "%s (%s)" % (self.layer.label, self.state.attribute.label)

    def get_layer_color(self):
        if self._viewer_state.color_mode == 'One color per layer':
            return self.state.color
        else:
            return self.state.cmap

    def enable(self):
        if hasattr(self, 'composite_image'):
            self.composite_image.invalidate_cache()
        super(ImageLayerArtist, self).enable()

    def remove(self):
        super(ImageLayerArtist, self).remove()
        if self.uuid in self.composite:
            self.composite.deallocate(self.uuid)
            self.composite_image.invalidate_cache()

    def get_image_shape(self):

        if self._viewer_state.x_att is None or self._viewer_state.y_att is None:
            return None

        x_axis = self._viewer_state.x_att.axis
        y_axis = self._viewer_state.y_att.axis

        full_shape = self._viewer_state.reference_data.shape

        return full_shape[y_axis], full_shape[x_axis]

    def get_image_data(self, view=None):

        try:
            image = self.state.get_sliced_data(view=view)
        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self.state.attribute)
            return None
        else:
            self.enable()

        return image

    def _update_image_data(self):
        self.composite_image.invalidate_cache()
        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        if not self.enabled:
            return

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

    @defer_draw
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
    def update(self, *event):
        self.state.reset_cache()
        self._update_image(force=True)
        self.redraw()


class ImageSubsetArray(object):

    def __init__(self, viewer_state, layer_artist):
        self._viewer_state = weakref.ref(viewer_state)
        self._layer_artist = weakref.ref(layer_artist)
        self._layer_state = weakref.ref(layer_artist.state)

    @property
    def layer_artist(self):
        return self._layer_artist()

    @property
    def layer_state(self):
        return self._layer_state()

    @property
    def viewer_state(self):
        return self._viewer_state()

    @property
    def shape(self):

        x_axis = self.viewer_state.x_att.axis
        y_axis = self.viewer_state.y_att.axis

        full_shape = self.viewer_state.reference_data.shape

        return full_shape[y_axis], full_shape[x_axis]

    def __getitem__(self, view=None):

        if (self.layer_artist is None or
                self.layer_state is None or
                self.viewer_state is None):
            return broadcast_to(np.nan, self.shape)

        # We should compute the mask even if the layer is not visible as we need
        # the layer to show up properly when it is made visible (which doesn't
        # trigger __getitem__)

        try:
            mask = self.layer_state.get_sliced_data(view=view)
        except IncompatibleAttribute:
            self.layer_artist.disable_incompatible_subset()
            return broadcast_to(np.nan, self.shape)
        else:
            self.layer_artist.enable()

        r, g, b = color2rgb(self.layer_state.color)
        mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = (255 * mask).astype(np.uint8)

        return mask

    @property
    def dtype(self):
        return np.uint8

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return np.product(self.shape)


class ImageSubsetLayerArtist(BaseImageLayerArtist):

    _layer_state_cls = ImageSubsetLayerState
    _python_exporter = python_export_image_subset_layer

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ImageSubsetLayerArtist, self).__init__(axes, viewer_state,
                                                     layer_state=layer_state, layer=layer)

        self.subset_array = ImageSubsetArray(self._viewer_state, self)

        self.image_artist = imshow(self.axes, self.subset_array,
                                   origin='lower', interpolation='nearest',
                                   vmin=0, vmax=1, aspect=self._viewer_state.aspect)

        self._line_x = self.axes.axvline(0)
        self._line_x.set_visible(False)

        self._line_y = self.axes.axhline(0)
        self._line_y.set_visible(False)

        self.mpl_artists = [self.image_artist, self._line_x, self._line_y]

    @defer_draw
    def _update_data(self):
        if isinstance(self.state.layer.subset_state, PixelSubsetState):
            slices = self.state.layer.subset_state.slices
            x = slices[self._viewer_state.x_att.axis].start
            y = slices[self._viewer_state.y_att.axis].start
            self._line_x.set_data([x, x], [0, 1])
            self._line_x.set_visible(True)
            self._line_y.set_data([0, 1], [y, y])
            self._line_y.set_visible(True)
        else:
            self._line_x.set_visible(False)
            self._line_y.set_visible(False)
        self.image_artist.invalidate_cache()
        self.redraw()  # forces subset to be recomputed

    @defer_draw
    def _update_visual_attributes(self):

        if not self.enabled:
            return

        for artist in self.mpl_artists:
            if artist is self.image_artist:
                artist.set_visible(self.state.visible)
                artist.set_alpha(self.state.alpha)
            else:
                if self.state.visible:
                    artist.set_visible(isinstance(self.state.layer.subset_state, PixelSubsetState))
                else:
                    artist.set_visible(False)
                artist.set_color(self.state.color)
                artist.set_alpha(self.state.alpha * 0.5)
            artist.set_zorder(self.state.zorder)

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

        if force or any(prop in changed for prop in ('layer', 'attribute', 'color',
                                                     'x_att', 'y_att', 'slices')):
            self._update_data()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('zorder', 'visible', 'alpha')):
            self._update_visual_attributes()

    def remove(self):
        super(ImageSubsetLayerArtist, self).remove()
        self.image_artist.invalidate_cache()

    def enable(self):
        super(ImageSubsetLayerArtist, self).enable()
        # We need to now ensure that image_artist, which may have been marked
        # as not being visible when the layer was cleared is made visible
        # again.
        if hasattr(self, 'image_artist'):
            self.image_artist.invalidate_cache()
            self._update_visual_attributes()

    @defer_draw
    def update(self, *event):
        self.state.reset_cache()
        self._update_image(force=True)
        self.redraw()
