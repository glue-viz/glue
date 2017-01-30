from __future__ import absolute_import, division, print_function

from glue.core.layer_artist import LayerArtistBase


# TODO: should use the built-in class for this, though we don't need
#       the _sync_style method, so just re-define here for now.


class MatplotlibLayerArtist(LayerArtistBase):

    def __init__(self, layer, axes, viewer_state):

        super(MatplotlibLayerArtist, self).__init__(layer)

        # Keep a reference to the layer (data or subset) and axes
        self.axes = axes
        self.viewer_state = viewer_state

        self.mpl_artists = []

    def clear(self):
        for artist in self.mpl_artists:
            try:
                artist.remove()
            except ValueError:  # already removed
                pass
        self.mpl_artists = []

    def redraw(self):
        self.axes.figure.canvas.draw()

    @property
    def zorder(self):
        return self._zorder

    @zorder.setter
    def zorder(self, value):
        self._zorder = value
        for artist in self.mpl_artists:
            artist.set_zorder(self._zorder)
        self.redraw()  # needed?

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        for artist in self.mpl_artists:
            artist.set_visible(self._visible)
        self.redraw()  # needed?
