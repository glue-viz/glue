import logging
import time

import numpy as np

from ..core.exceptions import IncompatibleAttribute
from ..core.util import color2rgb
from ..core.util import Pointer
from ..core.callback_property import (CallbackProperty)

from .image_client import ImageClient
from .layer_artist import LayerArtist

from ginga.util import wcsmod
wcsmod.use('astropy')
from ginga.ImageViewCanvas import Image
from ginga import AstroImage, RGBImage


class GingaClient(ImageClient):
    display_data = CallbackProperty(None)
    display_attribute = CallbackProperty(None)

    def __init__(self, data, canvas=None, artist_container=None):
        super(GingaClient, self).__init__(data, artist_container)
        self._setup_ginga(canvas)

    def _setup_ginga(self, canvas):

        if canvas is None:
            raise ValueError("GingaClient needs a canvas")

        self._canvas = canvas
        self._wcs = None

    def _new_rgb_layer(self, layer):
        pass

    def _new_subset_image_layer(self, layer):
        return GingaSubsetImageLayer(layer, self._canvas)

    def _new_image_layer(self, layer):
        return GingaImageLayer(layer, self._canvas)

    def _new_scatter_layer(self, layer):
        pass

    def _update_axis_labels(self):
        pass


class GingaLayerArtist(LayerArtist):
    zorder = Pointer('_zorder')
    visible = Pointer('_visible')

    def __init__(self, layer, canvas):
        super(GingaLayerArtist, self).__init__(layer, canvas)
        self._canvas = canvas
        self._visible = True

    def redraw(self):
        #pass
        print "ginga layer artist redraw"
        self._canvas.redraw()

    def _sync_style(self):
        pass


class GingaImageLayer(GingaLayerArtist):

    def __init__(self, layer, canvas):
        super(GingaImageLayer, self).__init__(layer, canvas)
        self._override_image = None
        self.norm = None  # XXX unused by Ginga, cleanup

    def set_norm(self, **kwargs):
        pass

    def override_image(self, image):
        """Temporarily show a different image"""
        #raise NotImplementedError()
        self._override_image = image

    def clear_override(self):
        self._override_image = None

    def clear(self):
        # how to clear image in ginga?
        pass

    def update(self, view, transpose=False):
        """
        to fix:
        view is downsampled/cropped. Let ginga do this
        check if we can skip this depending on attribute, data
        """

        self.clear()

        # TODO: check visibility

        if self._override_image != None:
            data = self.override_image
        else:
            data = self._layer[view]
            if transpose:
                data = data.T

        aimg = AstroImage.AstroImage(data_np=data)
        self._canvas.set_image(aimg)

        hdr = self._layer.coords._header
        aimg.update_keywords(hdr)


class GingaSubsetImageLayer(GingaLayerArtist):

    def __init__(self, layer, canvas):
        super(GingaSubsetImageLayer, self).__init__(layer, canvas)
        self._img = None
        self._cimg = None
        self._tag = "layer%s" % (str(layer.label))
        self._visible = True

    @property
    def visible(self):
        return self._visible
    
    @visible.setter
    def visible(self, value):
        self._visible = value
        print "subset visibility=%s" % (str(value))
        if not value:
            self.clear()
        elif self._cimg:
            self._canvas.add(self._cimg, tag=self._tag, redraw=True)
            
    def clear(self):
        try:
            self._canvas.deleteObjectsByTag([self._tag], redraw=True)
        except:
            pass

    def _compute_img(self, view, transpose=False):
        print "update subset image"
        time_start = time.time()
        subset = self.layer
        #self.clear()
        logging.debug("View into subset %s is %s", self.layer, view)
        print ("View into subset %s is %s", self.layer, view)
        id, ysl, xsl = view

        try:
            mask = subset.to_mask(view[1:])
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False
        logging.debug("View mask has shape %s", mask.shape)
        time_split = time.time()
        print "a) %.2f split time" % (time_split - time_start)

        # shortcut for empty subsets
        if not mask.any():
            return None

        if transpose:
            mask = mask.T
        time_split = time.time()
        print "b) %.2f split time" % (time_split - time_start)

        r, g, b = color2rgb(self.layer.style.color)

        time_split = time.time()
        print "c) %.2f split time" % (time_split - time_start)

        if self._img and self._img.get_data().shape[:2] == mask.shape[:2]:
            # optimization to simply update the color overlay if it already
            # exists and is the correct size
            data = self._img.get_data()
            data[..., 3] = 127 * mask
            return self._img

        # create new color image overlay
        ones = np.ones(mask.shape)
        clr_img = np.dstack((r * ones, g * ones, b * ones, mask * .5))
        clr_img = (255 * clr_img).astype(np.uint8)

        rgbimg = RGBImage.RGBImage(data_np=clr_img)

        self._img = rgbimg

        elapsed_time = time.time() - time_split
        print "%.2f sec to make color image" % (elapsed_time)
        return self._img

    def update(self, view, transpose=False):
        print("updating subset layer")
        # remove previously added image
        try:
            self._canvas.deleteObjectsByTag([self._tag], redraw=False)
        except:
            pass

        im = self._compute_img(view, transpose)
        if not im:
            return
        #print im.get_data()
        # lower z-order in the back
        # TODO: check for z-order
        
        x_pos = y_pos = 0
        self._cimg = Image(x_pos, y_pos, im, alpha=0.5,
                           flipy=False)
        if self._visible:
            self._canvas.add(self._cimg, tag=self._tag, redraw=True)

