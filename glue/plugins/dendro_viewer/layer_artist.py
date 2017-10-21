from __future__ import absolute_import, division, print_function

import numpy as np

from glue.utils import defer_draw

from glue.core.exceptions import IncompatibleAttribute
from glue.core.subset import Subset
# from glue.core.layer_artist import MatplotlibLayerArtist, ChangedTrigger
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.plugins.dendro_viewer.state import DendrogramLayerState


class DendrogramLayerArtist(MatplotlibLayerArtist):

    # X vertices of structure i are in layout[0][3*i: 3*i+3]
    # layout = ChangedTrigger()

    _layer_state_cls = DendrogramLayerState

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(DendrogramLayerArtist, self).__init__(axes, viewer_state,
                                                    layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update)
        self.state.add_global_callback(self._update)

        # TODO: following is temporary
        self.state.data_collection = self._viewer_state.data_collection
        self.data_collection = self._viewer_state.data_collection

        self.reset_cache()

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    @defer_draw
    def _update_dendrogram(self):

        self.remove()

        if self.state.viewer_state._layout is None:
            return

        # layout[0] is [x0, x0, x[parent0], nan, ...]
        # layout[1] is [y0, y[parent0], y[parent0], nan, ...]
        ids = 3 * np.arange(self.layer.data.size)

        try:
            if isinstance(self.layer, Subset):
                ids = ids[self.layer.to_mask()]
        except IncompatibleAttribute as exc:
            self.disable_invalid_attributes(*exc.args)
            return False

        x, y = self.state.viewer_state._layout.xy
        blank = np.zeros(ids.size) * np.nan
        x = np.column_stack([x[ids], x[ids + 1],
                             x[ids + 2], blank]).ravel()
        y = np.column_stack([y[ids], y[ids + 1],
                             y[ids + 2], blank]).ravel()

        self.mpl_artists = self.axes.plot(x, y, '-')

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

    @defer_draw
    def _update(self, force=False, **kwargs):

        if (self._viewer_state.height_att is None or
                self._viewer_state.parent_att is None or
                self._viewer_state.order_att is None or
                self.state.layer is None):
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to x_att changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_dendrogram is really only called once
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

        if force or any(prop in changed for prop in ('layer', 'height_att', 'parent_att', 'order_att')):
            self._update_dendrogram()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('linewidth', 'alpha', 'color', 'zorder', 'visible')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):

        # Recompute the histogram
        self._update(force=True)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
