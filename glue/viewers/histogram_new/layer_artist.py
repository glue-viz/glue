from __future__ import absolute_import, division, print_function

import numpy as np

from glue.utils import nonpartial

from glue.viewers.histogram_new.state import HistogramLayerState
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist


class HistogramLayerArtist(MatplotlibLayerArtist):

    def __init__(self, layer, axes, viewer_state, initial_layer_state=None):

        super(HistogramLayerArtist, self).__init__(layer, axes, viewer_state)

        # Set up a state object for the layer artist
        if initial_layer_state is None:
            initial = {}
        else:
            initial = initial_layer_state.as_dict()
            if 'layer' in initial:
                initial.pop('layer')

        # Set up a state object for the layer artist
        self.layer_state = HistogramLayerState(viewer_state=viewer_state, layer=layer, **initial)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.add_callback('*', nonpartial(self.update))
        self.viewer_state.add_callback('xatt', nonpartial(self.update))
        self.viewer_state.add_callback('log_x', nonpartial(self.update))
        self.viewer_state.add_callback('log_y', nonpartial(self.update))
        self.viewer_state.add_callback('cumulative', nonpartial(self.update))
        self.viewer_state.add_callback('normalize', nonpartial(self.update))
        self.viewer_state.add_callback('hist_x_min', nonpartial(self.update))
        self.viewer_state.add_callback('hist_x_max', nonpartial(self.update))
        self.viewer_state.add_callback('hist_n_bin', nonpartial(self.update))

        self.layer_state.add_callback('*', nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

    def update(self):

        if self.viewer_state.hist_x_min is None or self.viewer_state.hist_x_max is None:
            return

        x = self.layer[self.viewer_state.xatt]
        x = x[~np.isnan(x) & (x >= self.viewer_state.hist_x_min) & (x <= self.viewer_state.hist_x_max)]

        # TODO: is there a better way to do this?
        self.clear()

        if len(x) == 0:
            self.redraw()
            return

        # For histogram
        xmin, xmax = sorted([self.viewer_state.hist_x_min, self.viewer_state.hist_x_max])
        if self.viewer_state.log_x:
            range = None
            bins = np.logspace(np.log10(xmin), np.log10(xmax), self.viewer_state.hist_n_bin)
        else:
            range = [xmin, xmax]
            bins = self.viewer_state.hist_n_bin

        result = self.axes.hist(x, range=range,
                                bins=bins,
                                zorder=self.zorder,
                                edgecolor='black',
                                facecolor=self.layer_state.color,
                                alpha=self.layer_state.alpha,
                                cumulative=self.viewer_state.cumulative,
                                normed=self.viewer_state.normalize)

        self.mpl_hist = result[0]
        self.mpl_bins = result[1]
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

        if self.viewer_state.log_y:
            self.viewer_state.y_min = result[0][result[0] > 0].min() / 10
        else:
            self.viewer_state.y_min = 0

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
