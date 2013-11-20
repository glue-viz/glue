"""
LayerArtist classes handle the visualization of an individual subset
or dataset
"""
import logging

import numpy as np
from matplotlib.cm import gray
from ..core.exceptions import IncompatibleAttribute
from ..core.util import color2rgb
from ..core.subset import Subset
from .util import view_cascade, get_extent
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


class LayerArtist(object):

    def __init__(self, layer, axes):
        """Create a new LayerArtist

        :param layer: Data or subset to draw
        :type layer: :class:`~glue.core.data.Data` or `glue.core.subset.Subset`
        """
        self.layer = layer
        self._axes = axes
        self._visible = True

        self._zorder = 0
        self.view = None
        self.artists = []

        self._changed = True
        self._state = None  # cache of subset state, if relevant

    def redraw(self):
        self._axes.figure.canvas.draw()

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        for artist in self.artists:
            artist.set_zorder(value)
        self._zorder = value

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        for a in self.artists:
            a.set_visible(value)

    @property
    def enabled(self):
        return len(self.artists) > 0

    def update(self, view=None):
        """Redraw this layer"""
        raise NotImplementedError()

    def clear(self):
        """Clear the visulaization for this layer"""
        for artist in self.artists:
            try:
                artist.remove()
            except ValueError:  # already removed
                pass
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
        for artist in self.artists:
            artist.set_markeredgecolor('none')
            artist.set_markerfacecolor(style.color)
            artist.set_marker(style.marker)
            artist.set_markersize(style.markersize)
            artist.set_linestyle('None')
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)

    def __str__(self):
        return "%s for %s" % (self.__class__.__name__, self.layer.label)

    __repr__ = __str__


class ImageLayerArtist(LayerArtist):

    def __init__(self, layer, ax):
        super(ImageLayerArtist, self).__init__(layer, ax)
        self.norm = None
        self._cmap = gray

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
            result.vmin = vals[.01 * vals.size]
            result.vmax = vals[.99 * vals.size]
        return result

    def update(self, view):
        self.clear()
        views = view_cascade(self.layer, view)
        artists = []

        lr0 = self.layer[views[0]]
        self.norm = self.norm or self._default_norm(lr0)
        self.norm = self.norm or self._default_norm(lr0)
        self.norm.update_clip(self.layer, view[0])

        for v in views:
            image = self.layer[v]
            extent = get_extent(v)
            artists.append(self._axes.imshow(image, cmap=self.cmap,
                                             norm=self.norm,
                                             interpolation='nearest',
                                             origin='lower',
                                             extent=extent, zorder=0))
        self.artists = artists
        self._sync_style()

    def set_norm(self, vmin=None, vmax=None,
                 bias=None, contrast=None, stretch=None, norm=None,
                 clip_lo=None, clip_hi=None):
        if norm is not None:
            self.norm = norm
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


class RGBImageLayerArtist(ImageLayerArtist):

    def __init__(self, layer, ax):
        super(RGBImageLayerArtist, self).__init__(layer, ax)
        self.r = None
        self.g = None
        self.b = None
        self.rnorm = None
        self.gnorm = None
        self.bnorm = None
        self.contrast_layer = 'green'
        self.layer_visible = dict(red=True, green=True, blue=True)

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

    def update(self, view=None):
        self.clear()
        if self.r is None or self.g is None or self.b is None:
            return

        if view is None:
            view = self._last_view

        if view is None:
            return
        self._last_view = view

        views = view_cascade(self.layer, view)
        artists = []
        for v in views:
            extent = get_extent(v)
            # first argument = component. swap
            r = tuple([self.r] + list(v[1:]))
            g = tuple([self.g] + list(v[1:]))
            b = tuple([self.b] + list(v[1:]))
            r = self.layer[r]
            g = self.layer[g]
            b = self.layer[b]
            self.rnorm = self.rnorm or self._default_norm(r)
            self.gnorm = self.gnorm or self._default_norm(g)
            self.bnorm = self.bnorm or self._default_norm(b)
            if v is views[0]:
                self.rnorm.update_clip(self.layer, self.r)
                self.gnorm.update_clip(self.layer, self.g)
                self.bnorm.update_clip(self.layer, self.b)

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
        self.artists = artists
        self._sync_style()


class SubsetImageLayerArtist(LayerArtist):

    def update(self, view):
        subset = self.layer
        self.clear()
        logging.debug("View into subset %s is %s", self.layer, view)

        try:
            mask = subset.to_mask(view[1:])
        except IncompatibleAttribute:
            return
        logging.debug("View mask has shape %s", mask.shape)

        # shortcut for empty subsets
        if not mask.any():
            return

        extent = get_extent(view)
        r, g, b = color2rgb(self.layer.style.color)
        mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = (255 * mask).astype(np.uint8)
        self.artists = [self._axes.imshow(mask, extent=extent,
                                          interpolation='nearest',
                                          origin='lower',
                                          zorder=5, visible=self.visible)]


class ScatterLayerArtist(LayerArtist):
    xatt = ChangedTrigger()
    yatt = ChangedTrigger()

    def __init__(self, layer, ax):
        super(ScatterLayerArtist, self).__init__(layer, ax)
        self.emphasis = None

    def _recalc(self):
        self.clear()
        assert len(self.artists) == 0

        try:
            x = self.layer[self.xatt].ravel()
            y = self.layer[self.yatt].ravel()
        except IncompatibleAttribute:
            return False
        self.artists = self._axes.plot(x, y)
        return True

    def update(self, view=None):
        self._check_subset_state_changed()
        if self._changed:
            if not self._recalc():
                return
            self._changed = False

        has_emph = False
        if self.emphasis is not None:
            try:
                s = Subset(self.layer.data)
                s.subset_state = self.emphasis & self.layer.subset_state
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
        """Remove a LayerArtist from this collection"""
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
        return iter(self.artists)

    def __contains__(self, item):
        if isinstance(item, LayerArtist):
            return item in self.artists
        return any(item is a.layer for a in self.artists)

    def __getitem__(self, layer):
        if isinstance(layer, int):
            return self.artists[layer]
        return [a for a in self.artists if a.layer is layer]


class HistogramLayerArtist(LayerArtist):
    lo = ChangedTrigger(0)
    hi = ChangedTrigger(1)
    nbins = ChangedTrigger(10)
    xlog = ChangedTrigger(False)

    def __init__(self, layer, axes):
        super(HistogramLayerArtist, self).__init__(layer, axes)
        self.ylog = False
        self.cumulative = False
        self.normed = False
        self.y = np.array([])
        self.x = np.array([])
        self._y = np.array([])

        self._scale_state = None

    def has_patches(self):
        return len(self.artists) > 0

    def get_data(self):
        return self.x, self.y

    def clear(self):
        super(HistogramLayerArtist, self).clear()
        self.x = np.array([])
        self.y = np.array([])
        self._y = np.array([])

    def _calculate_histogram(self):
        """Recalculate the histogram, creating new patches"""
        self.clear()
        try:
            data = self.layer[self.att].ravel()
        except IncompatibleAttribute:
            return False

        if data.size == 0:
            return

        if self.lo > np.nanmax(data) or self.hi < np.nanmin(data):
            return
        if self.xlog:
            data = np.log10(data)
            rng = [np.log10(self.lo), np.log10(self.hi)]
        else:
            rng = self.lo, self.hi
        nbinpatch = self._axes.hist(data,
                                    bins=self.nbins,
                                    range=rng)
        self._y, self.x, self.artists = nbinpatch
        return True

    def _scale_histogram(self):
        """Modify height of bins to match ylog, cumulative, and norm"""
        if self.x.size == 0:
            return

        y = self._y.astype(np.float)
        dx = self.x[1] - self.x[0]
        if self.normed:
            div = y.sum() * dx
            if div == 0:
                div = 1
            y /= div
        if self.cumulative:
            y = y.cumsum()
            y /= y.max()

        self.y = y
        bottom = 0 if not self.ylog else 1e-100

        for a, y in zip(self.artists, y):
            a.set_height(y)
            x, y = a.get_xy()
            a.set_xy((x, bottom))

    def _check_scale_histogram(self):
        """
        If needed, rescale histogram to match cumulative/log/normed state.
        """
        state = (self.normed, self.ylog, self.cumulative)
        if state == self._scale_state:
            return
        self._scale_state = state
        self._scale_histogram()

    def update(self, view=None):
        """Sync plot.

        The _change flag tracks whether the histogram needs to be
        recalculated. If not, the properties of the existing
        artists are updated
        """
        self._check_subset_state_changed()
        if self._changed:
            if not self._calculate_histogram():
                return
            self._changed = False
            self._scale_state = None
        self._check_scale_histogram()
        self._sync_style()

    def _sync_style(self):
        """Update visual properties"""
        style = self.layer.style
        for artist in self.artists:
            artist.set_facecolor(style.color)
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)
