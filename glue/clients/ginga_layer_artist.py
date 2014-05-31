"""
LayerArtist classes handle the visualization of an individual subset
or dataset
"""
import logging

import numpy as np
from matplotlib.cm import gray

from ginga import AstroImage, RGBImage, LayerImage

from ..core.exceptions import IncompatibleAttribute
from ..core.util import color2rgb, PropertySetMixin, Pointer
from ..core.subset import Subset
from .util import view_cascade, get_extent, small_view, small_view_array
from .ds9norm import DS9Normalize

class ChangedTrigger(object):

    """Sets an instance's _changed attribute to True on update"""

    def __init__(self, default=None):
        self._default = default
        self._vals = {}

    def __get__(self, inst, type=None):
        return self._vals.get(inst, self._default)

    def __set__(self, inst, value):
        changed = value != self.__get__(inst)
        self._vals[inst] = value
        if changed:
            inst._changed = True


class LayerArtist(PropertySetMixin):
    _property_set = ['zorder', 'visible', 'layer']

    def __init__(self, layer, canvas):
        """Create a new LayerArtist

        :param layer: Data or subset to draw
        :type layer: :class:`~glue.core.data.Data` or `glue.core.subset.Subset`
        """
        self._layer = layer
        self._canvas = canvas
        self._visible = True

        self._zorder = 0
        self.view = None
        self.artists = []

        self._changed = True
        self._state = None  # cache of subset state, if relevant
        self._disabled_reason = ''

    def disable(self, reason):
        self._disabled_reason = reason
        self.clear()

    def disable_invalid_attributes(self, *attributes):
        if len(attributes) == 0:
            self.disable('')

        msg = ('Layer depends on attributes that '
               'cannot be derived for %s:\n -%s' %
               (self._layer.data.label,
                '\n -'.join(map(str, attributes))))

        self.disable(msg)

    @property
    def disabled_message(self):
        if self.enabled:
            return ''
        return "Cannot visualize this layer\n%s" % self._disabled_reason

    def redraw(self):
        self._canvas.redraw()

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, value):
        self._layer = value

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        ## for artist in self.artists:
        ##     artist.set_zorder(value)
        self._zorder = value

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        ## for a in self.artists:
        ##     a.set_visible(value)

    @property
    def enabled(self):
        return len(self.artists) > 0

    def update(self, view=None):
        """Redraw this layer"""
        raise NotImplementedError()

    def clear(self):
        """Clear the visualization for this layer"""
        self.artists = []

    def _check_subset_state_changed(self):
        """Checks to see if layer is a subset and, if so,
        if it has changed subset state. Sets _changed flag to True if so"""
        if not isinstance(self.layer, Subset):
            return
        state = self.layer.subset_state
        if state is not self._state:
            self._changed = True
            self._state = state

    def _sync_style(self):
        style = self.layer.style
        ## for artist in self.artists:
        ##     edgecolor = style.color if style.marker == '+' else 'none'
        ##     artist.set_markeredgecolor(edgecolor)
        ##     artist.set_markeredgewidth(3)
        ##     artist.set_markerfacecolor(style.color)
        ##     artist.set_marker(style.marker)
        ##     artist.set_markersize(style.markersize)
        ##     artist.set_linestyle('None')
        ##     artist.set_alpha(style.alpha)
        ##     artist.set_zorder(self.zorder)
        ##     artist.set_visible(self.visible and self.enabled)

    def __str__(self):
        return "%s for %s" % (self.__class__.__name__, self.layer.label)

    def __gluestate__(self, context):
        # note, this doesn't yet have a restore method. Will rely on client
        return dict((k, context.id(v)) for k, v in self.properties.items())

    __repr__ = __str__


class ImageLayerArtist(LayerArtist):
    _property_set = LayerArtist._property_set + ['norm']

    def __init__(self, layer, canvas):
        super(ImageLayerArtist, self).__init__(layer, canvas)
        self._norm = None
        self._cmap = gray
        self._override_image = None
        self._clip_cache = None

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
        ## for a in self.artists:
        ##     a.set_cmap(value)

    def redraw(self):
        self._canvas.redraw()

    def _default_norm(self, layer):
        vals = np.sort(layer.ravel())
        vals = vals[np.isfinite(vals)]
        result = DS9Normalize()
        result.stretch = 'arcsinh'
        result.clip = True
        if vals.size > 0:
            result.vmin = vals[.01 * vals.size]
            result.vmax = vals[.99 * vals.size]
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

    def update(self, view, transpose=False):
        print "view is", view
        self.clear()
        #views = view_cascade(self.layer, view)
        artists = []

        ## lr0 = self._extract_view(views[0], transpose)
        ## self.norm = self.norm or self._default_norm(lr0)
        ## self.norm = self.norm or self._default_norm(lr0)
        ## self._update_clip(views[0][0])

        ## for v in views:
        ##     image = self._extract_view(v, transpose)
        ##     extent = get_extent(v, transpose)
                
        image = self._extract_view(view, transpose)
        # TODO: need metadata
        aimg = AstroImage.AstroImage(data_np=image)
        artists.append(aimg)
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
        ## for artist in self.artists:
        ##     artist.set_zorder(self.zorder)
        ##     artist.set_visible(self.visible and self.enabled)
        pass


class RGBImageLayerArtist(ImageLayerArtist):
    _property_set = ImageLayerArtist._property_set + \
        ['r', 'g', 'b', 'rnorm', 'gnorm', 'bnorm', 'color_visible']

    r = ChangedTrigger()
    g = ChangedTrigger()
    b = ChangedTrigger()
    rnorm = Pointer('_rnorm')
    gnorm = Pointer('_gnorm')
    bnorm = Pointer('_bnorm')

    def __init__(self, layer, ax, last_view=None):
        super(RGBImageLayerArtist, self).__init__(layer, ax)
        self.contrast_layer = 'green'
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

    @property
    def color_visible(self):
        return [self.layer_visible['red'], self.layer_visible['green'],
                self.layer_visible['blue']]

    @color_visible.setter
    def color_visible(self, value):
        self.layer_visible['red'] = value[0]
        self.layer_visible['green'] = value[1]
        self.layer_visible['blue'] = value[2]

    def update(self, view=None, transpose=False):
        self.clear()
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

            cimg = RGBImage.RGBImage(data_np=image)
            artists.append(cimg)
            
        self.artists = artists
        self._sync_style()


class SubsetImageLayerArtist(LayerArtist):

    def update(self, view, transpose=False):
        print "update subset image"
        subset = self.layer
        self.clear()
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
        #mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = np.dstack((r * mask, g * mask, b * mask))
        mask = (255 * mask).astype(np.uint8)

        print "making cimg"
        cimg = RGBImage.RGBImage(data_np=mask)
        print "made cimg"
        self.artists = [cimg]


class ScatterLayerArtist(LayerArtist):
    xatt = ChangedTrigger()
    yatt = ChangedTrigger()
    _property_set = LayerArtist._property_set + ['xatt', 'yatt']

    def __init__(self, layer, ax):
        super(ScatterLayerArtist, self).__init__(layer, ax)
        self.emphasis = None  # an optional SubsetState of emphasized points

    def _recalc(self):
        self.clear()
        assert len(self.artists) == 0

        try:
            x = self.layer[self.xatt].ravel()
            y = self.layer[self.yatt].ravel()
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        self.artists = self._axes.plot(x, y)
        return True

    def update(self, view=None, transpose=False):
        self._check_subset_state_changed()

        if self._changed:  # erase and make a new artist
            if not self._recalc():  # no need to update style
                return
            self._changed = False

        has_emph = False
        if self.emphasis is not None:
            try:
                s = Subset(self.layer.data)
                s.subset_state = self.emphasis
                if hasattr(self.layer, 'subset_state'):
                    s.subset_state &= self.layer.subset_state
                x = s[self.xatt].ravel()
                y = s[self.yatt].ravel()
                self.artists.extend(self._axes.plot(x, y))
                has_emph = True
            except IncompatibleAttribute:
                pass

        self._sync_style()
        if has_emph:
            self.artists[-1].set_mec('green')
            self.artists[-1].set_mew(2)
            self.artists[-1].set_alpha(1)

    def get_data(self):
        try:
            return self.layer[self.xatt].ravel(), self.layer[self.yatt].ravel()
        except IncompatibleAttribute:
            return np.array([]), np.array([])


class LayerArtistContainer(object):

    """A collection of LayerArtists"""

    def __init__(self):
        self.artists = []

    def _duplicate(self, artist):
        for a in self.artists:
            if type(a) == type(artist) and a.layer is artist.layer:
                return True
        return False

    def _check_duplicate(self, artist):
        """Raise an error if this artist is a duplicate"""
        if self._duplicate(artist):
            raise ValueError("Already have an artist for this type "
                             "and data")

    def append(self, artist):
        """Add a LayerArtist to this collection"""
        self._check_duplicate(artist)
        self.artists.append(artist)
        artist.zorder = max(a.zorder for a in self.artists) + 1

    def remove(self, artist):
        """Remove a LayerArtist from this collection

        :param artist: The artist to remove
        :type artist: :class:`LayerArtist`
        """
        try:
            self.artists.remove(artist)
            artist.clear()
        except ValueError:
            pass

    def pop(self, layer):
        """Remove all artists associated with a layer"""
        to_remove = [a for a in self.artists if a.layer is layer]
        for r in to_remove:
            self.remove(r)
        return to_remove

    @property
    def layers(self):
        """A list of the unique layers in the container"""
        return list(set([a.layer for a in self.artists]))

    def __len__(self):
        return len(self.artists)

    def __iter__(self):
        return iter(sorted(self.artists, key=lambda x: x.zorder))

    def __contains__(self, item):
        if isinstance(item, LayerArtist):
            return item in self.artists
        return any(item is a.layer for a in self.artists)

    def __getitem__(self, layer):
        if isinstance(layer, int):
            return self.artists[layer]
        return [a for a in self.artists if a.layer is layer]


