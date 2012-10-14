"""
LayerArtist classes handle the visualization of an individual subset
or dataset
"""
import logging

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import gray
from ..core.exceptions import IncompatibleAttribute
from ..core.util import color2rgb
from .util import view_cascade, get_extent


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
        return self._visible and self.enabled

    @visible.setter
    def visible(self, value):
        self._visible = value
        for a in self.artists:
            a.set_visible(value)

    @property
    def enabled(self):
        return len(self.artists) > 0

    def update(self):
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
        result = InvNormalize()
        result.vmin = vals[.05 * vals.size]
        result.vmax = vals[.95 * vals.size]
        return result

    def update(self, view):
        self.clear()
        views = view_cascade(self.layer, view)
        artists = []
        for v in views:
            image = self.layer[v]
            extent = get_extent(v)
            self.norm = self.norm or self._default_norm(image)
            artists.append(self._axes.imshow(image, cmap=self.cmap,
                                             norm=self.norm,
                                             interpolation='nearest',
                                             origin='lower',
                                             extent=extent, zorder=0))
        self.artists = artists

    def set_norm(self, vmin, vmax):
        if vmin is not None:
            self.norm.vmin = vmin
        if vmax is not None:
            self.norm.vmax = vmax


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

        #shortcut for empty subsets
        if not mask.any():
            return

        extent = get_extent(view)
        r, g, b = color2rgb(self.layer.style.color)
        mask = np.dstack((r * mask, g * mask, b * mask, mask * .5))
        mask = (255 * mask).astype(np.uint8)
        self.artists = [self._axes.imshow(mask, extent=extent,
                                          interpolation='nearest',
                                          origin='lower',
                                          zorder=5)]


class ScatterLayerArtist(LayerArtist):
    def __init__(self, layer, ax):
        super(ScatterLayerArtist, self).__init__(layer, ax)
        self.xatt = None
        self.yatt = None

    def update(self):
        self.clear()
        assert len(self.artists) == 0
        try:
            x = self.layer[self.xatt].ravel()
            y = self.layer[self.yatt].ravel()
        except IncompatibleAttribute:
            return
        self.artists = self._axes.plot(x, y)
        self._sync_style()

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


class InvNormalize(Normalize):
    """ Simple wrapper to matplotlib Normalize object, that
    handles the case where vmax <= vmin """
    def __call__(self, value):
        if self.vmax <= self.vmin:
            self.vmax, self.vmin = self.vmin, self.vmax
            result = 1 - Normalize.__call__(self, value)
            self.vmax, self.vmin = self.vmin, self.vmax
        else:
            result = Normalize.__call__(self, value)
        return result
