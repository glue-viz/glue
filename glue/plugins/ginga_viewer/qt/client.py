from __future__ import absolute_import, division, print_function

import logging
from time import time

import numpy as np
from ginga.misc import Bunch
from ginga.util import wcsmod
from ginga import AstroImage, BaseImage

from glue.core.util import split_component_view
from glue.core.exceptions import IncompatibleAttribute
from glue.core.layer_artist import LayerArtistBase
from glue.utils import view_shape, stack_view, color2rgb, Pointer

from glue.viewers.image.client import ImageClient
from glue.viewers.image.layer_artist import ImageLayerBase, SubsetImageLayerBase

wcsmod.use('astropy')


class GingaClient(ImageClient):

    def __init__(self, data, canvas=None, layer_artist_container=None):
        super(GingaClient, self).__init__(data, layer_artist_container)
        self._setup_ginga(canvas)

    def _setup_ginga(self, canvas):

        if canvas is None:
            raise ValueError("GingaClient needs a canvas")

        self._canvas = canvas
        self._wcs = None
        self._crosshair_id = '_crosshair'

    def _new_rgb_layer(self, layer):
        raise NotImplementedError()

    def _new_subset_image_layer(self, layer):
        return GingaSubsetImageLayer(layer, self._canvas)

    def _new_image_layer(self, layer):
        return GingaImageLayer(layer, self._canvas)

    def _new_scatter_layer(self, layer):
        raise NotImplementedError()

    def _update_axis_labels(self):
        pass

    def _update_and_redraw(self):
        pass

    def set_cmap(self, cmap):
        self._canvas.set_cmap(cmap)

    def show_crosshairs(self, x, y):
        self.clear_crosshairs()
        c = self._canvas.viewer.getDrawClass('point')(x, y, 6, color='red',
                                                      style='plus')
        self._canvas.add(c, tag=self._crosshair_id, redraw=True)

    def clear_crosshairs(self):
        try:
            self._canvas.deleteObjectsByTag([self._crosshair_id], redraw=False)
        except:
            pass


class GingaLayerArtist(LayerArtistBase):

    zorder = Pointer('_zorder')
    visible = Pointer('_visible')

    def __init__(self, layer, canvas):
        super(GingaLayerArtist, self).__init__(layer)
        self._canvas = canvas
        self._visible = True

    def redraw(self, whence=0):
        self._canvas.redraw(whence=whence)


class GingaImageLayer(GingaLayerArtist, ImageLayerBase):

    # unused by Ginga
    cmap = None
    norm = None

    def __init__(self, layer, canvas):
        super(GingaImageLayer, self).__init__(layer, canvas)
        self._override_image = None
        self._tag = "layer%s_%s" % (layer.label, time())
        self._img = None  # DataImage instance
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
        elif self._img:
            self._canvas.set_image(self._img)

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        self._zorder = value
        try:
            canvas_img = self._canvas.getObjectByTag('_image')
            canvas_img.set_zorder(value)
        except KeyError:
            # object does not yet exist on canvas
            pass

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
        if not self.visible:
            return

        # update ginga model
        comp, view = split_component_view(view)

        if self._img is None:
            self._img = DataImage(self.layer, comp, view, transpose)
            self._canvas.set_image(self._img)

        self._img.data = self.layer
        self._img.component = comp
        self._img.view = view
        self._img.transpose = transpose
        self._img.override_image = self._override_image

        self.redraw()


class GingaSubsetImageLayer(GingaLayerArtist, SubsetImageLayerBase):

    def __init__(self, layer, canvas):
        super(GingaSubsetImageLayer, self).__init__(layer, canvas)
        self._img = None
        self._cimg = None
        self._tag = "layer%s_%s" % (layer.label, time())
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

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        self._zorder = value
        try:
            canvas_img = self._canvas.getObjectByTag(self._tag)
            canvas_img.set_zorder(value)
        except KeyError:
            # object does not yet exist on canvas
            pass

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
            # SubsetImages can't be added to canvases directly. Need
            # to wrap into a ginga canvas type.
            Image = self._canvas.getDrawClass('image')
            self._cimg = Image(0, 0, self._img, alpha=0.5, flipy=False)

        self._img.view = view
        self._img.color = (r, g, b)
        self._img.transpose = transpose

    def _check_enabled(self):
        """
        Sync the enabled/disabled status, based on whether
        mask is computable
        """
        self._enabled = True
        try:
            # the first pixel
            view = tuple(0 for _ in self.layer.data.shape)
            self.layer.to_mask(view)
        except IncompatibleAttribute as exc:
            self._enabled = False
            self.disable_invalid_attributes(*exc.args)
        return self._enabled

    def _ensure_added(self):
        """ Add artist to canvas if needed """
        try:
            self._canvas.getObjectByTag(self._tag)
        except KeyError:
            self._canvas.add(self._cimg, tag=self._tag, redraw=False)

    def update(self, view, transpose=False):

        self._check_enabled()
        self._update_ginga_models(view, transpose)

        if self._enabled and self._visible:
            self._ensure_added()
        else:
            self.clear()

        self.redraw(whence=0)


def forbidden(*args):
    raise ValueError("Forbidden")


class DataImage(AstroImage.AstroImage):

    """
    A Ginga image subclass to interface with Glue Data objects
    """
    get_data = _get_data = copy_data = set_data = get_array = transfer = forbidden

    def __init__(self, data, component, view, transpose=False,
                 override_image=None, **kwargs):
        """
        Parameters
        ----------
        data : glue.core.data.Data
            The data to image
        component : glue.core.data.ComponentID
            The ComponentID in the data to image
        view : numpy-style view
            The view into the data to image. Must produce a 2D array
        transpose : bool
            Whether to transpose the view
        override_image : numpy array, optional
            Whether to show override_image instead of the view into the data.
            The override image must have the same shape as the 2D view into
            the data.
        kwargs : dict
            Extra kwargs are passed to the superclass
        """
        self.transpose = transpose
        self.view = view
        self.data = data
        self.component = component
        self.override_image = None
        super(DataImage, self).__init__(**kwargs)

    @property
    def shape(self):
        """
        The shape of the 2D view into the data
        """
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
        if self.override_image is not None:
            return self.override_image[view]

        # Combining multiple views: First a 2D slice into an ND array, then
        # the requested view from this slice
        if self.transpose:
            views = [self.view, 'transpose', view]
        else:
            views = [self.view, view]
        view = stack_view(self.data.shape, *views)
        return self.data[self.component, view]


class SubsetImage(BaseImage.BaseImage):

    """
    A Ginga image subclass to interface with Glue subset objects
    """
    get_data = _get_data = copy_data = set_data = get_array = transfer = forbidden

    def __init__(self, subset, view, color=(0, 1, 0), transpose=False, **kwargs):
        """
        Parameters
        ----------
        subset : glue.core.subset.Subset
            The subset to image
        view : numpy-style view
            The view into the subset to image. Must produce a 2D array
        color : tuple of 3 floats in range [0, 1]
            The color to image the subset as
        transpose : bool
            Whether to transpose the view
        kwargs : dict
            Extra kwargs are passed to the ginga superclass
        """
        super(SubsetImage, self).__init__(**kwargs)
        self.subset = subset
        self.view = view
        self.transpose = transpose
        self.color = color
        self.order = 'RGBA'

    @property
    def shape(self):
        """
        Shape of the 2D view into the subset mask
        """
        result = view_shape(self.subset.data.shape, self.view)
        if self.transpose:
            result = result[::-1]
        return tuple(list(result) + [4])  # 4th dim is RGBA channels

    def _rgb_from_mask(self, mask):
        """
        Turn a boolean mask into a 4-channel RGBA image
        """
        r, g, b = self.color
        ones = mask * 0 + 255
        alpha = mask * 127
        result = np.dstack((ones * r, ones * g, ones * b, alpha)).astype(np.uint8)
        return result

    def _get_fast_data(self):
        return self._slice((slice(None, None, 10), slice(None, None, 10)))

    def _slice(self, view):
        """
        Extract a view from the 2D subset mask.
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
        # we already know the data bounds
        self.minval = 0
        self.maxval = 256
        self.minval_noinf = self.minval
        self.maxval_noinf = self.maxval

    def get_scaled_cutout_wdht(self, x1, y1, x2, y2, new_wd, new_ht):
        doit = getattr(self, '_doit', False)
        self._doit = not doit
        # default implementation if downsampling
        if doit or new_wd <= (x2 - x1 + 1) or new_ht <= (y2 - y1 + 1):
            return super(SubsetImage, self).get_scaled_cutout_wdht(x1, y1, x2, y2, new_wd, new_ht)

        # if upsampling, prevent extra to_mask() computation
        x1, x2 = np.clip([x1, x2], 0, self.width - 2).astype(np.int)
        y1, y2 = np.clip([y1, y2], 0, self.height - 2).astype(np.int)

        result = self._slice(np.s_[y1:y2 + 1, x1:x2 + 1])

        yi = np.linspace(0, result.shape[0], new_ht).astype(np.int).reshape(-1, 1).clip(0, result.shape[0] - 1)
        xi = np.linspace(0, result.shape[1], new_wd).astype(np.int).reshape(1, -1).clip(0, result.shape[1] - 1)
        yi, xi = [np.array(a) for a in np.broadcast_arrays(yi, xi)]
        result = result[yi, xi]

        scale_x = 1.0 * result.shape[1] / (x2 - x1 + 1)
        scale_y = 1.0 * result.shape[0] / (y2 - y1 + 1)

        return Bunch.Bunch(data=result, scale_x=scale_x, scale_y=scale_y)
