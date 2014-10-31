import logging
from time import time

import numpy as np

from ..core.exceptions import IncompatibleAttribute
from ..core.util import Pointer, view_shape, stack_view, split_component_view, color2rgb

from .image_client import ImageClient
from .ds9norm import DS9Normalize
from .layer_artist import (ChangedTrigger, LayerArtist, RGBImageLayerBase,
                           ImageLayerBase, SubsetImageLayerBase)

from ginga.util import wcsmod
wcsmod.use('astropy')
from ginga.ImageViewCanvas import Image
from ginga import AstroImage, BaseImage


class GingaClient(ImageClient):

    def __init__(self, data, canvas=None, artist_container=None):
        super(GingaClient, self).__init__(data, artist_container)
        self._setup_ginga(canvas)

    def _setup_ginga(self, canvas):

        if canvas is None:
            raise ValueError("GingaClient needs a canvas")

        self._canvas = canvas
        self._wcs = None

    def _new_rgb_layer(self, layer):
        return RGBGingaImageLayer(layer, self._canvas)

    def _new_subset_image_layer(self, layer):
        return GingaSubsetImageLayer(layer, self._canvas)

    def _new_image_layer(self, layer):
        return GingaImageLayer(layer, self._canvas)

    def _new_scatter_layer(self, layer):
        pass

    def _update_axis_labels(self):
        pass

    def set_cmap(self, cmap):
        self._canvas.set_cmap(cmap)


class GingaLayerArtist(LayerArtist):
    zorder = Pointer('_zorder')
    visible = Pointer('_visible')

    def __init__(self, layer, canvas):
        # Note: a bit ugly here, canvas gets assigned to self._axes
        #       by superclass. This doesn't actually do anything harmful
        #       right now, but it's a hack.
        super(GingaLayerArtist, self).__init__(layer, canvas)
        self._canvas = canvas
        self._visible = True

    def redraw(self):
        self._canvas.redraw()

    def _sync_style(self):
        pass


class GingaImageLayer(GingaLayerArtist, ImageLayerBase):

    # unused by Ginga
    cmap = None
    norm = None

    def __init__(self, layer, canvas):
        super(GingaImageLayer, self).__init__(layer, canvas)
        self._override_image = None
        self._tag = "layer%s_%s" % (layer.label, time())
        self._img = None
        self._aimg = None
        self._enabled = True

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        if self._visible == value:
            return

        self._visible = value
        if not value:
            self.clear()
        elif self._aimg:
            self._canvas.set_image(self._aimg)

    def set_norm(self, **kwargs):
        # NOP for ginga
        pass

    def clear_norm(self):
        # NOP for ginga
        pass

    def override_image(self, image):
        """Temporarily show a different image"""
        self._override_image = image

    def clear_override(self):
        self._override_image = None

    def clear(self):
        # remove previously added image
        try:
            self._canvas.deleteObjectsByTag(['_image'], redraw=False)
        except:
            pass

    @property
    def enabled(self):
        return self._enabled

    def update(self, view, transpose=False):
        """
        to fix:
        view is downsampled/cropped. Let ginga do this
        check if we can skip this depending on attribute, data
        """
        if not self.visible:
            return

        comp, view = split_component_view(view)

        if self._aimg is None:
            self._aimg = DataImage(self.layer, comp, view, transpose)
            self._canvas.set_image(self._aimg)

        self._aimg.data = self.layer
        self._aimg.component = comp
        self._aimg.view = view
        self._aimg.transpose = transpose
        self._aimg.override_image = self.override_image

        self.redraw()


def forbidden(*args):
    raise ValueError("Forbidden")


class DataImage(AstroImage.AstroImage):
    get_data = _get_data = copy_data = set_data = get_array = transfer = forbidden

    def __init__(self, data, component, view, transpose=False,
                 override_image=None, **kwargs):
        self.transpose = transpose
        self.view = view
        self.data = data
        self.component = component
        self.override_image = None
        super(DataImage, self).__init__(**kwargs)

    @property
    def shape(self):
        result = view_shape(self.data.shape, self.view)
        if self.transpose:
            result = result[::-1]
        return result

    def _get_fast_data(self):
        return self._slice((slice(None, None, 10), slice(None, None, 10)))

    def _slice(self, view):
        """
        Extract a view from the 2D image.
        """
        # Combining multiple views: First a 2D slice into an ND array, then
        # the requested view from this slice

        if self.transpose:
            views = [self.view, 'transpose', view]
        else:
            views = [self.view, view]
        view = stack_view(self.data.shape, *views)
        return self.data[self.component, view]


class SubsetImage(BaseImage.BaseImage):
    get_data = _get_data = copy_data = set_data = get_array = transfer = forbidden

    def __init__(self, subset, view, color=(0, 1, 0), transpose=False, **kwargs):
        super(SubsetImage, self).__init__(**kwargs)
        self.subset = subset
        self.view = view
        self.transpose = transpose
        self.color = color
        self.order = 'RGBA'

    @property
    def shape(self):
        result = view_shape(self.subset.data.shape, self.view)
        if self.transpose:
            result = result[::-1]
        return tuple(list(result) + [4])  # 4th dim is RGBA channels

    def _rgb_from_mask(self, mask):

        r, g, b = self.color
        ones = mask * 0 + 255
        alpha = mask * 127
        result = np.dstack((ones * r, ones * g, ones * b, alpha)).astype(np.uint8)
        return result

    def _get_fast_data(self):
        return self._slice((slice(None, None, 10), slice(None, None, 10)))

    def _slice(self, view):
        """
        Extract a view from the 2D image.
        """
        # Combining multiple views: First a 2D slice into an ND array, then
        # the requested view from this slice

        if self.transpose:
            views = [self.view, 'transpose', view]
        else:
            views = [self.view, view]
        view = stack_view(self.subset.data.shape, *views)

        mask = self.subset.to_mask(view)
        return self._rgb_from_mask(mask)

    def _set_minmax(self):
        self.minval = 0
        self.maxval = 256
        self.minval_noinf = self.minval
        self.maxval_noinf = self.maxval


class GingaSubsetImageLayer(GingaLayerArtist, SubsetImageLayerBase):

    def __init__(self, layer, canvas):
        super(GingaSubsetImageLayer, self).__init__(layer, canvas)
        self._img = None
        self._cimg = None
        self._tag = "layer%s_%s" % (layer.label, time())
        self._visible = True
        self._enabled = True

    @property
    def visible(self):
        return self._visible

    @property
    def enabled(self):
        return self._enabled

    @visible.setter
    def visible(self, value):
        if value is self._visible:
            return

        self._visible = value
        if not value:
            self.clear()
        elif self._cimg:
            self._canvas.add(self._cimg, tag=self._tag, redraw=True)

    def clear(self):
        try:
            self._canvas.deleteObjectsByTag([self._tag], redraw=True)
        except:
            pass

    def _update_ginga_models(self, view, transpose=False):
        subset = self.layer
        logging.getLogger(__name__).debug("View into subset %s is %s", self.layer, view)

        _, view = split_component_view(view)  # discard ComponentID
        r, g, b = color2rgb(self.layer.style.color)

        if self._img is None:
            self._img = SubsetImage(subset, view)
        if self._cimg is None:
            self._cimg = Image(0, 0, self._img, alpha=0.5, flipy=False)

        self._img.view = view
        self._img.color = (r, g, b)
        self._img.transpose = transpose

        return True

    def _check_enabled(self):
        self._enabled = True
        try:
            view = tuple(0 for _ in self.layer.data.shape)
            self.layer.to_mask(view)
        except IncompatibleAttribute as exc:
            self._enabled = False
            self.disable_invalid_attributes(*exc.args)

    def update(self, view, transpose=False):
        self.clear()

        self._check_enabled()
        self._update_ginga_models(view, transpose)

        if self._enabled and self._visible:
            self._canvas.add(self._cimg, tag=self._tag, redraw=False)

        self.redraw()


class RGBGingaImageLayer(GingaLayerArtist, RGBImageLayerBase):
    r = ChangedTrigger(None)
    g = ChangedTrigger(None)
    b = ChangedTrigger(None)

    rnorm = gnorm = bnorm = None

    contrast_layer = Pointer('_contrast_layer')
    layer_visible = Pointer('_layer_visible')

    def __init__(self, layer, canvas, last_view=None):
        super(RGBGingaImageLayer, self).__init__(layer, canvas)
        self.contrast_layer = 'green'
        self.layer_visible = dict(red=True, green=True, blue=True)
        self._aimg = None

    @property
    def norm(self):
        return getattr(self, self.contrast_layer[0] + 'norm')

    @norm.setter
    def norm(self, value):
        setattr(self, self.contrast_layer[0] + 'norm', value)

    def set_norm(self, **kwargs):
        norm = self.norm or DS9Normalize()

        for k, v in kwargs:
            setattr(norm, k, v)

        self.norm = norm

    def update(self, view=None, transpose=None):
        self.clear()

        rgb = []
        shp = self.layer.shape
        for att, norm, ch in zip([self.r, self.g, self.b],
                                 [self.rnorm, self.gnorm, self.bnorm],
                                 ['red', 'green', 'blue']):
            if att is None or not self.layer_visible[ch]:
                rgb.append(np.zeros(shp))
                continue

            data = self.layer[att]
            norm = norm or DS9Normalize()
            data = norm(data)

            rgb.append(data)

        self._aimg = AstroImage.AstroImage(data_np=np.dstack(rgb))
        hdr = self._layer.coords._header
        self._aimg.update_keywords(hdr)

        if self._visible:
            self._canvas.set_image(self._aimg)
