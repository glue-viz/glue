from __future__ import absolute_import, division, print_function

import numpy as np

from glue.utils import nonpartial

from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist

__all__ = ['ScatterLayerArtist']


class ScatterLayerArtist(MatplotlibLayerArtist):

    def __init__(self, layer, axes, viewer_state):

        super(ScatterLayerArtist, self).__init__(layer, axes, viewer_state)

        # Set up a state object for the layer artist
        self.layer_state = ScatterLayerState(layer=layer)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.connect_all(nonpartial(self.update))
        self.viewer_state.connect('xatt', nonpartial(self.update))
        self.viewer_state.connect('yatt', nonpartial(self.update))

        self.layer_state.connect_all(nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

        # Set up an initially empty artist
        self.scatter_artist = self.axes.scatter([], [])
        self.plot_artist = self.axes.plot([], [], 'o', mec='none')[0]

        self.mpl_artists = self.scatter_artist, self.plot_artist

    def update(self):

        x = self.layer[self.viewer_state.xatt[0]]
        y = self.layer[self.viewer_state.yatt[0]]

        # TODO: is there a better way to do this?
        # self.clear()

        if self.layer_state.size_mode == 'Fixed' and self.layer_state.color_mode == 'Fixed':

            # In this case we use Matplotlib's plot function because it has much
            # better performance than scatter.

            offsets = np.dstack(([], []))
            self.scatter_artist.set_offsets(offsets)

            self.plot_artist.set_data(x, y)
            self.plot_artist.set_color(self.layer_state.color)
            self.plot_artist.set_markersize(self.layer_state.size * self.layer_state.size_scaling)
            self.plot_artist.set_alpha(self.layer_state.alpha)

        else:

            if self.layer_state.color_mode == 'Fixed':
                c = self.layer_state.color
                vmin = vmax = cmap = None
            else:
                c = self.layer[self.layer_state.cmap_attribute[0]]
                vmin = self.layer_state.cmap_vmin
                vmax = self.layer_state.cmap_vmax
                cmap = self.layer_state.cmap

            if self.layer_state.size_mode == 'Fixed':
                s = self.layer_state.size
            else:
                s = self.layer[self.layer_state.size_attribute[0]]
                s = ((s - self.layer_state.size_vmin) /
                     (self.layer_state.size_vmax - self.layer_state.size_vmin)) * 100

            s *= self.layer_state.size_scaling

            self.plot_artist.set_data([], [])

            offsets = np.dstack((x, y))

            self.scatter_artist.set_offsets(offsets)

            if self.layer_state.size_mode == 'Fixed':
                s = np.broadcast_to(s, x.shape)

            # Note, we need to square here because for scatter, s is actually
            # proportional to the marker area, not radius.
            self.scatter_artist.set_sizes(s ** 2)

            if self.layer_state.color_mode == 'Fixed':
                self.scatter_artist.set_facecolors(c)
            else:
                self.scatter_artist.set_array(c)
                self.scatter_artist.set_cmap(cmap)
                self.scatter_artist.set_clim(vmin, vmax)

            self.scatter_artist.set_edgecolor('none')

            self.scatter_artist.set_zorder(self.zorder)
            self.scatter_artist.set_alpha(self.layer_state.alpha)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
