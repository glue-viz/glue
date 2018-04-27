from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core import Data
from glue.utils import defer_draw, nanmin, nanmax
from glue.viewers.profile.state import ProfileLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute


class ProfileLayerArtist(MatplotlibLayerArtist):

    _layer_state_cls = ProfileLayerState

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ProfileLayerArtist, self).__init__(axes, viewer_state,
                                                 layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_profile)
        self.state.add_global_callback(self._update_profile)

        self.plot_artist = self.axes.plot([1, 2, 3], [3, 4, 5], 'k-', drawstyle='steps-mid')[0]

        self.mpl_artists = [self.plot_artist]

        self.reset_cache()

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    @defer_draw
    def _calculate_profile(self):

        x, y = self.state.profile

        if x is None or y is None:
            self.disable_invalid_attributes(self._viewer_state.x_att)
            return
        else:
            self.enable()

        self._visible_data = x, y

        # Update the data values.
        if len(x) > 0:
            # Normalize profile values to the [0:1] range based on limits
            if self._viewer_state.normalize:
                y = self.state.normalize_values(y)
            self.plot_artist.set_data(x, y)
            self.plot_artist.set_visible(True)
        else:
            # We need to do this otherwise we get issues on Windows when
            # passing an empty list to plot_artist
            self.plot_artist.set_visible(False)

        if len(x) == 0:
            return

        # TODO: the following was copy/pasted from the histogram viewer, maybe
        # we can find a way to avoid duplication?

        # We have to do the following to make sure that we reset the y_max as
        # needed. We can't simply reset based on the maximum for this layer
        # because other layers might have other values, and we also can't do:
        #
        #   self._viewer_state.y_max = max(self._viewer_state.y_max, result[0].max())
        #
        # because this would never allow y_max to get smaller.

        if not self._viewer_state.normalize:

            self.state._y_min = nanmin(y)
            self.state._y_max = nanmax(y) * 1.2

            largest_y_max = max(getattr(layer, '_y_max', 0) for layer in self._viewer_state.layers)
            if largest_y_max != self._viewer_state.y_max:
                self._viewer_state.y_max = largest_y_max

            smallest_y_min = min(getattr(layer, '_y_min', np.inf) for layer in self._viewer_state.layers)
            if smallest_y_min != self._viewer_state.y_min:
                self._viewer_state.y_min = smallest_y_min

        self.redraw()

    @defer_draw
    def _update_visual_attributes(self):

        if not self.enabled:
            return

        for mpl_artist in self.mpl_artists:
            mpl_artist.set_visible(self.state.visible)
            mpl_artist.set_zorder(self.state.zorder)
            mpl_artist.set_color(self.state.color)
            mpl_artist.set_alpha(self.state.alpha)
            mpl_artist.set_linewidth(self.state.linewidth)

        self.redraw()

    def _update_profile(self, force=False, **kwargs):

        # TODO: we need to factor the following code into a common method.

        if (self._viewer_state.x_att is None or
                self.state.attribute is None or
                self.state.layer is None):
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to x_att changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_profile is really only called once
        # then we could consider simplifying this. Until then, we manually keep track
        # of which properties have changed.

        changed = set()

        if not force:

            for key, value in self._viewer_state.as_dict().items():
                if value != self._last_viewer_state.get(key, None):
                    changed.add(key)

            for key, value in self.state.as_dict().items():
                if value != self._last_layer_state.get(key, None):
                    changed.add(key)

        self._last_viewer_state.update(self._viewer_state.as_dict())
        self._last_layer_state.update(self.state.as_dict())

        if force or any(prop in changed for prop in ('layer', 'x_att', 'attribute', 'function', 'normalize', 'v_min', 'v_max')):
            self._calculate_profile()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('alpha', 'color', 'zorder', 'visible', 'linewidth')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):
        try:
            self.state._update_profile()
        except IncompatibleAttribute:
            if isinstance(self.state.layer, Data):
                self.disable_invalid_attributes(self.state.attribute)
            else:
                self.disable_incompatible_subset()
        else:
            self.enable()
        self._update_profile(force=True)
        self.redraw()
