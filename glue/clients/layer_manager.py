"""
LayerManager classes handle the visualization of an individual subset
or dataset
"""
import numpy as np

from ..core.exceptions import IncompatibleAttribute


class LayerManager(object):
    def __init__(self, layer, axes):
        """Create a new LayerManager

        :param layer: Data or subset to draw
        :type layer: :class:`~glue.core.data.Data` or `glue.core.subset.Subset`
        """
        self._layer = layer
        self._axes = axes
        self._visible = True

        self.zorder = 0
        self.view = None
        self.artists = []

    @property
    def visible(self):
        return self._visible and self.enabled

    @visible.setter
    def visible(self, value):
        self._visible = value

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
        style = self._layer.style
        for artist in self.artists:
            artist.set_markeredgecolor('none')
            artist.set_markerfacecolor(style.color)
            artist.set_marker(style.marker)
            artist.set_markersize(style.markersize)
            artist.set_linestyle('None')
            artist.set_alpha(style.alpha)
            artist.set_zorder(self.zorder)
            artist.set_visible(self.visible and self.enabled)


class ImageLayerManager(LayerManager):
    def __init__(self, layer, ax):
        super(ImageLayerManager, self).__init__(layer, ax)
        self.norm = None


class ScatterLayerManager(LayerManager):
    def __init__(self, layer, ax):
        super(ScatterLayerManager, self).__init__(layer, ax)
        self.xatt = None
        self.yatt = None

    def update(self):
        self.clear()
        assert len(self.artists) == 0
        try:
            x = self._layer[self.xatt].ravel()
            y = self._layer[self.yatt].ravel()
        except IncompatibleAttribute:
            return
        self.artists = self._axes.plot(x, y)
        self._sync_style()

    def get_data(self):
        try:
            return self._layer[self.xatt], self._layer[self.yatt]
        except IncompatibleAttribute:
            return np.array([]), np.array([])


class MaskLayerManager(LayerManager):
    pass
