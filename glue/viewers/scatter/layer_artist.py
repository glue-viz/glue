from __future__ import absolute_import, division, print_function

from glue.utils import defer_draw

from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute


class ScatterLayerArtist(MatplotlibLayerArtist):

    _layer_state_cls = ScatterLayerState

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ScatterLayerArtist, self).__init__(axes, viewer_state,
                                                 layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_scatter)
        self.state.add_global_callback(self._update_scatter)

        # TODO: following is temporary
        self.state.data_collection = self._viewer_state.data_collection
        self.data_collection = self._viewer_state.data_collection

        self.mpl_artists = self.axes.plot([], [], 'o', mec='none')

        self.reset_cache()

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    def _update_scatter_data(self):

        if len(self.mpl_artists) == 0:
            return

        try:
            x = self.layer[self._viewer_state.x_att].ravel()
        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self._viewer_state.x_att)
            return
        else:
            self._enabled = True

        try:
            y = self.layer[self._viewer_state.y_att].ravel()
        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self._viewer_state.y_att)
            return
        else:
            self._enabled = True

        self.mpl_artists[0].set_data(x, y)

    @defer_draw
    def _update_visual_attributes(self):

        for mpl_artist in self.mpl_artists:
            mpl_artist.set_visible(self.state.visible)
            mpl_artist.set_zorder(self.state.zorder)
            mpl_artist.set_markeredgecolor('none')
            mpl_artist.set_markerfacecolor(self.state.color)
            mpl_artist.set_markersize(self.state.size)
            mpl_artist.set_alpha(self.state.alpha)

        self.redraw()

    def _update_scatter(self, force=False, **kwargs):

        if (self._viewer_state.x_att is None or
            self._viewer_state.y_att is None or
            self.state.layer is None):
            return

        # Figure out which attributes are different from before. Ideally we shouldn't
        # need this but currently this method is called multiple times if an
        # attribute is changed due to x_att changing then hist_x_min, hist_x_max, etc.
        # If we can solve this so that _update_histogram is really only called once
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

        if force or any(prop in changed for prop in ('layer', 'x_att', 'y_att')):
            self._update_scatter_data()
            force = True  # make sure scaling and visual attributes are updated

        if force or any(prop in changed for prop in ('size', 'alpha', 'color', 'zorder', 'visible')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):

        self._update_scatter(force=True)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
