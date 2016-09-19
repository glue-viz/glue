from __future__ import absolute_import, division, print_function

import numpy as np

from glue.utils import nonpartial

from glue.viewers.histogram.state import HistogramLayerState
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist


class HistogramLayerArtist(MatplotlibLayerArtist):

    def __init__(self, layer, axes, viewer_state):

        super(HistogramLayerArtist, self).__init__(layer, axes, viewer_state)

        # Set up a state object for the layer artist
        self.layer_state = HistogramLayerState(layer=layer)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.connect_all(nonpartial(self.update))
        self.viewer_state.connect('xatt', nonpartial(self.update))
        self.viewer_state.connect('n_bins', nonpartial(self.update))
        self.viewer_state.connect('x_min', nonpartial(self.update))
        self.viewer_state.connect('x_max', nonpartial(self.update))
        self.viewer_state.connect('cumulative', nonpartial(self.update))
        self.viewer_state.connect('normalize', nonpartial(self.update))

        self.layer_state.connect_all(nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

    def update(self):

        x = self.layer[self.viewer_state.xatt[0]]
        x = x[~np.isnan(x) & (x >= self.viewer_state.x_min) & (x <= self.viewer_state.x_max)]

        # TODO: is there a better way to do this?
        self.clear()

        if len(x) == 0:
            return

        # For histogram
        result = self.axes.hist(x, range=sorted([self.viewer_state.x_min,
                                                 self.viewer_state.x_max]),
                                bins=self.viewer_state.n_bins,
                                zorder=self.zorder,
                                edgecolor='black',
                                facecolor=self.layer_state.color,
                                alpha=self.layer_state.alpha,
                                cumulative=self.viewer_state.cumulative,
                                normed=self.viewer_state.normalize)

        self.mpl_artists = result[2]

        # We have to do the following to make sure that we reset the y_max as
        # needed. We can't simply reset based on the maximum for this layer
        # because other layers might have other values, and we also can't do:
        #
        #   self.viewer_state.y_max = max(self.viewer_state.y_max, result[0].max())
        #
        # because this would never allow y_max to get smaller.

        self.layer_state._y_max = result[0].max()

        if self.viewer_state.log_y:
            self.layer_state._y_max *= 2
        else:
            self.layer_state._y_max *= 1.2

        for layer in self.viewer_state.layers:
            if self.layer_state != layer and hasattr(layer, '_y_max') and self.layer_state._y_max < layer._y_max:
                break
        else:
            self.viewer_state.y_max = self.layer_state._y_max

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
