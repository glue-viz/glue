from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np
from matplotlib.cm import gray

from glue.external import six
from glue.core.exceptions import IncompatibleAttribute
from glue.core.layer_artist import MatplotlibLayerArtist, ChangedTrigger
from glue.core.util import small_view, small_view_array
from glue.utils import view_cascade, get_extent, color2rgb, Pointer

from .ds9norm import DS9Normalize

__all__ = ['RGBImageLayerArtist', 'ImageLayerArtist']


@six.add_metaclass(ABCMeta)
class RGBImageLayerBase(object):

    r = abstractproperty()               # ComponentID for red channel
    g = abstractproperty()               # ComponentID for green channel
    b = abstractproperty()               # ComponentID for blue channel
    rnorm = abstractproperty()           # Normalize instance for red channel
    gnorm = abstractproperty()           # Normalize instance for green channel
    bnorm = abstractproperty()  # Normalize instance for blue channel
    contrast_layer = abstractproperty()  # 'red' | 'green' | 'blue'. Which norm to adjust during set_norm
    layer_visible = abstractproperty()   # dict (str->bool). Whether to show 'red', 'green', 'blue' layers

    @property
    def color_visible(self):
        """
        Return layer visibility as a list of [red_visible, green_visible, blue_visible]
        """
        return [self.layer_visible['red'], self.layer_visible['green'],
                self.layer_visible['blue']]

    @color_visible.setter
    def color_visible(self, value):
        self.layer_visible['red'] = value[0]
        self.layer_visible['green'] = value[1]
        self.layer_visible['blue'] = value[2]


@six.add_metaclass(ABCMeta)
class ImageLayerBase(object):

    norm = abstractproperty()  # Normalization instance to scale intensities
    cmap = abstractproperty()  # colormap

    @abstractmethod
    def set_norm(self, **kwargs):
        """
        Adjust the normalization instance parameters.
        See :class:`glue.viewers.image.ds9norm.DS9Normalize attributes for valid
        kwargs for this function
        """
        pass

    @abstractmethod
    def clear_norm():
        """
        Reset the norm to the default
        """
        pass

    @abstractmethod
    def override_image(self, image):
        """
        Temporarily display another image instead of a view into the data

        The new image has the same shape as the view into the data
        """
        pass

    @abstractmethod
    def clear_override(self):
        """
        Remove the override image, and display the data again
        """
        pass


@six.add_metaclass(ABCMeta)
class SubsetImageLayerBase(object):
    pass


class ImageLayerArtist(MatplotlibLayerArtist, ImageLayerBase):
    _property_set = MatplotlibLayerArtist._property_set + ['norm']

    def __init__(self, layer, ax):
        super(ImageLayerArtist, self).__init__(layer, ax)
        self._norm = None
        self._cmap = gray
        self._override_image = None
        self._clip_cache = None
        self.aspect = 'equal'

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        self._cmap = value
        for a in self.artists:
            a.set_cmap(value)

    def _default_norm(self, layer):
        vals = np.sort(layer.ravel())
        vals = vals[np.isfinite(vals)]
        result = DS9Normalize()
        result.stretch = 'arcsinh'
        result.clip = True
        if vals.size > 0:
            result.vmin = vals[np.intp(.01 * vals.size)]
            result.vmax = vals[np.intp(.99 * vals.size)]
        return result

    def override_image(self, image):
        """Temporarily show a different image"""
        self._override_image = image

    def clear_override(self):
        self._override_image = None

    def _extract_view(self, view, transpose):
        if self._override_image is None:
            result = self.layer[view]
            if transpose:
                result = result.T
            return result
        else:
            v = [v for v in view if isinstance(v, slice)]
            if transpose:
                v = v[::-1]
            result = self._override_image[v]
            return result

    def _update_clip(self, att):
        key = (att, id(self._override_image),
               self.norm.clip_lo, self.norm.clip_hi)
        if self._clip_cache == key:
            return
        self._clip_cache = key

        if self._override_image is None:
            data = small_view(self.layer, att)
        else:
            data = small_view_array(self._override_image)
        self.norm.update_clip(data)

    def update(self, view, transpose=False, aspect=None):

        if aspect is not None:
            self.aspect = aspect

        self.clear()
        views = view_cascade(self.layer, view)
        artists = []

        lr0 = self._extract_view(views[0], transpose)
        self.norm = self.norm or self._default_norm(lr0)
        self.norm = self.norm or self._default_norm(lr0)
        self._update_clip(views[0][0])

        for v in views:
            image = self._extract_view(v, transpose)
            extent = get_extent(v, transpose)
            artists.append(self._axes.imshow(image, cmap=self.cmap,
                                             norm=self.norm,
                                             interpolation='nearest',
                                             origin='lower',
                                             extent=extent, zorder=0))
            self._axes.set_aspect(self.aspect, adjustable='datalim')
        self.artists = artists
        self._sync_style()

    def set_norm(self, vmin=None, vmax=None,
                 bias=None, contrast=None, stretch=None, norm=None,
                 clip_lo=None, clip_hi=None):
        if norm is not None:
            self.norm = norm  # XXX Should wrap ala DS9Normalize(norm)
            return norm
        if self.norm is None:
            self.norm = DS9Normalize()
        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax
        if bias is not None:
            self.norm.bias = bias
        if contrast is not None:
            self.norm.contrast = contrast
        if clip_lo is not None:
            self.norm.clip_lo = clip_lo
        if clip_hi is not None:
            self.norm.clip_hi = clip_hi
        if stretch is not None:
            self.norm.stretch = stretch
        return self.norm

    def clear_norm(self):
        self.norm = None

    def _sync_style(self):
        for artist in self.artists:
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)


class RGBImageLayerArtist(ImageLayerArtist, RGBImageLayerBase):
    _property_set = ImageLayerArtist._property_set + \
        ['r', 'g', 'b', 'rnorm', 'gnorm', 'bnorm', 'color_visible']

    r = ChangedTrigger()
    g = ChangedTrigger()
    b = ChangedTrigger()
    rnorm = Pointer('_rnorm')
    gnorm = Pointer('_gnorm')
    bnorm = Pointer('_bnorm')

    # dummy class-level variables will be masked
    # at instance level, needed for ABC to be happy
    layer_visible = None
    contrast_layer = None

    def __init__(self, layer, ax, last_view=None):
        super(RGBImageLayerArtist, self).__init__(layer, ax)
        self.contrast_layer = 'green'
        self.aspect = 'equal'
        self.layer_visible = dict(red=True, green=True, blue=True)
        self.last_view = last_view

    def set_norm(self, *args, **kwargs):
        spr = super(RGBImageLayerArtist, self).set_norm
        if self.contrast_layer == 'red':
            self.norm = self.rnorm
            self.rnorm = spr(*args, **kwargs)
        if self.contrast_layer == 'green':
            self.norm = self.gnorm
            self.gnorm = spr(*args, **kwargs)
        if self.contrast_layer == 'blue':
            self.norm = self.bnorm
            self.bnorm = spr(*args, **kwargs)

    def update(self, view=None, transpose=False, aspect=None):

        self.clear()

        if aspect is not None:
            self.aspect = aspect

        if self.r is None or self.g is None or self.b is None:
            return

        if view is None:
            view = self.last_view

        if view is None:
            return
        self.last_view = view

        views = view_cascade(self.layer, view)
        artists = []
        for v in views:
            extent = get_extent(v, transpose)
            # first argument = component. swap
            r = tuple([self.r] + list(v[1:]))
            g = tuple([self.g] + list(v[1:]))
            b = tuple([self.b] + list(v[1:]))
            r = self.layer[r]
            g = self.layer[g]
            b = self.layer[b]
            if transpose:
                r = r.T
                g = g.T
                b = b.T
            self.rnorm = self.rnorm or self._default_norm(r)
            self.gnorm = self.gnorm or self._default_norm(g)
            self.bnorm = self.bnorm or self._default_norm(b)
            if v is views[0]:
                self.rnorm.update_clip(small_view(self.layer, self.r))
                self.gnorm.update_clip(small_view(self.layer, self.g))
                self.bnorm.update_clip(small_view(self.layer, self.b))

            image = np.dstack((self.rnorm(r),
                               self.gnorm(g),
                               self.bnorm(b)))

            if not self.layer_visible['red']:
                image[:, :, 0] *= 0
            if not self.layer_visible['green']:
                image[:, :, 1] *= 0
            if not self.layer_visible['blue']:
                image[:, :, 2] *= 0

            artists.append(self._axes.imshow(image,
                                             interpolation='nearest',
                                             origin='lower',
                                             extent=extent, zorder=0))
            self._axes.set_aspect(self.aspect, adjustable='datalim')
        self.artists = artists
        self._sync_style()


class SubsetImageLayerArtist(MatplotlibLayerArtist, SubsetImageLayerBase):

    def __init__(self, *args, **kwargs):
        super(SubsetImageLayerArtist, self).__init__(*args, **kwargs)
        self.aspect = 'equal'

    def update(self, view, transpose=False, aspect=None):

        self.clear()

        if aspect is not None:
            self.aspect = aspect

        subset = self.layer
        logging.debug("View into subset %s is %s", self.layer, view)

        try:
            mask = subset.to_mask(view[1:])
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False
        logging.debug("View mask has shape %s", mask.shape)

        # shortcut for empty subsets
        if not mask.any():
            return

        if transpose:
            mask = mask.T

        extent = get_extent(view, transpose)
        r, g, b = color2rgb(self.layer.style.color)
        mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = (255 * mask).astype(np.uint8)
        self.artists = [self._axes.imshow(mask, extent=extent,
                                          interpolation='nearest',
                                          origin='lower',
                                          zorder=5, visible=self.visible)]
        self._axes.set_aspect(self.aspect, adjustable='datalim')
