from __future__ import absolute_import, division, print_function

import numpy as np

from glue.utils import defer_draw

from glue.viewers.histogram_new.state import HistogramLayerState
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist


class HistogramLayerArtist(MatplotlibLayerArtist):

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(HistogramLayerArtist, self).__init__(layer, axes, viewer_state)

        self.layer = layer or layer_state.layer

        # Set up a state object for the layer artist
        self.state = layer_state or HistogramLayerState(viewer_state=viewer_state, layer=self.layer)
        self.viewer_state.layers.append(self.state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self.viewer_state.add_callback('*', self._update_histogram, as_kwargs=True)
        self.state.add_callback('*', self._update_histogram, as_kwargs=True)

        # TODO: following is temporary
        self.state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

        self.reset_cache()

    def clear(self):
        super(HistogramLayerArtist, self).clear()
        self.mpl_hist_unscaled = np.array([])
        self.mpl_hist = np.array([])
        self.mpl_bins = np.array([])

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    @defer_draw
    def _calculate_histogram(self):

        self.clear()

        x = self.layer[self.viewer_state.xatt]
        x = x[~np.isnan(x) & (x >= self.viewer_state.hist_x_min) & (x <= self.viewer_state.hist_x_max)]

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

        self.mpl_hist_unscaled, self.mpl_bins, self.mpl_artists = self.axes.hist(x, range=range, bins=bins)

    @defer_draw
    def _scale_histogram(self):

        if self.mpl_bins.size == 0 or self.mpl_hist_unscaled.sum() == 0:
            return

        self.mpl_hist = self.mpl_hist_unscaled.astype(np.float)
        dx = self.mpl_bins[1] - self.mpl_bins[0]

        if self.viewer_state.cumulative:
            self.mpl_hist = self.mpl_hist.cumsum()
            if self.viewer_state.normalize:
                self.mpl_hist /= self.mpl_hist.max()
        elif self.viewer_state.normalize:
            self.mpl_hist /= (self.mpl_hist.sum() * dx)

        bottom = 0 if not self.viewer_state.log_y else 1e-100

        for mpl_artist, y in zip(self.mpl_artists, self.mpl_hist):
            mpl_artist.set_height(y)
            x, y = mpl_artist.get_xy()
            mpl_artist.set_xy((x, bottom))

        # We have to do the following to make sure that we reset the y_max as
        # needed. We can't simply reset based on the maximum for this layer
        # because other layers might have other values, and we also can't do:
        #
        #   self.viewer_state.y_max = max(self.viewer_state.y_max, result[0].max())
        #
        # because this would never allow y_max to get smaller.

        self.state._y_max = self.mpl_hist.max()

        if self.viewer_state.log_y:
            self.state._y_max *= 2
        else:
            self.state._y_max *= 1.2

        for layer in self.viewer_state.layers:
            if self.state != layer and hasattr(layer, '_y_max') and self.state._y_max < layer._y_max:
                break
        else:
            self.viewer_state.y_max = self.state._y_max

        if self.viewer_state.log_y:
            self.viewer_state.y_min = self.mpl_hist[self.mpl_hist > 0].min() / 10
        else:
            self.viewer_state.y_min = 0

        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        for mpl_artist in self.mpl_artists:
            mpl_artist.set_visible(self.state.visible)
            mpl_artist.set_zorder(self.state.zorder)
            mpl_artist.set_edgecolor('none')
            mpl_artist.set_facecolor(self.state.color)
            mpl_artist.set_alpha(self.state.alpha)

        self.redraw()

    def _update_histogram(self, force=False, **kwargs):

        if (self.viewer_state.hist_x_min is None or
            self.viewer_state.hist_x_max is None or
            self.viewer_state.hist_n_bin is None or
            self.viewer_state.xatt is None or
            self.state.layer is None):
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to xatt changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_histogram is really only called once
        # then we could consider simplifying this. Until then, we manually keep track
        # of which properties have changed.

        changed = set()

        if not force:

            for key, value in self.viewer_state.as_dict().items():
                if value != self._last_viewer_state.get(key, None):
                    changed.add(key)

            for key, value in self.state.as_dict().items():
                if value != self._last_layer_state.get(key, None):
                    changed.add(key)

        self._last_viewer_state.update(self.viewer_state.as_dict())
        self._last_layer_state.update(self.state.as_dict())

        if force or any(prop in changed for prop in ('layer', 'xatt', 'hist_x_min', 'hist_x_max', 'hist_n_bin', 'log_x')):
            self._calculate_histogram()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('log_y', 'normalize', 'cumulative')):
            self._scale_histogram()

        if force or any(prop in changed for prop in ('alpha', 'color', 'zorder', 'visible')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):

        # Recompute the histogram
        self._update_histogram(force=True)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
