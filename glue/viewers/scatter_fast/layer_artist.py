from __future__ import absolute_import, division, print_function

import numpy as np

from glue.utils import nonpartial

from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist

from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from raster_axes.raster_axes import RasterizedScatter


__all__ = ['FastScatterLayerArtist']


class FastScatterLayerArtist(MatplotlibLayerArtist):

    def __init__(self, layer, axes, viewer_state):

        super(FastScatterLayerArtist, self).__init__(layer, axes, viewer_state)

        # Set up a state object for the layer artist
        self.layer_state = ScatterLayerState(layer=layer)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.connect_all(nonpartial(self.update))
        self.viewer_state.add_callback('xatt', nonpartial(self.update))
        self.viewer_state.add_callback('yatt', nonpartial(self.update))

        self.layer_state.add_callback('*', nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

        # Set up an initially empty artist
        self.norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())

        e = np.array([])
        self.scatter_artist = RasterizedScatter(self.axes, e, e, norm=self.norm)

        self.mpl_artists = [self.scatter_artist]

    @property
    def zorder(self):
        return 0

    @zorder.setter
    def zorder(self, value):
        return

    def update(self):

        x = self.layer[self.viewer_state.xatt[0]]
        y = self.layer[self.viewer_state.yatt[0]]

        # TODO: is there a better way to do this?
        # self.clear()

        self.scatter_artist.x = x.astype(float).ravel()
        self.scatter_artist.y = y.astype(float).ravel()
        self.scatter_artist._update(None)
        self.scatter_artist.set(color=self.layer_state.color, alpha=self.layer_state.alpha)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
