from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib.colors import Normalize

from glue.utils import defer_draw, broadcast_to
from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute

CMAP_PROPERTIES = set(['cmap_mode', 'cmap_att', 'cmap_vmin', 'cmap_vmax', 'cmap'])
SIZE_PROPERTIES = set(['size_mode', 'size_att', 'size_vmin', 'size_vmax', 'size_scaling', 'size'])
LINE_PROPERTIES = set(['linewidth', 'linestyle'])
VISUAL_PROPERTIES = (CMAP_PROPERTIES | SIZE_PROPERTIES |
                     LINE_PROPERTIES | set(['color', 'alpha', 'zorder', 'visible']))

DATA_PROPERTIES = set(['layer', 'x_att', 'y_att', 'cmap_mode', 'size_mode',
                       'xerr_att', 'yerr_att', 'xerr_visible', 'yerr_visible'])


class InvertedNormalize(Normalize):
    def __call__(self, *args, **kwargs):
        return 1 - super(InvertedNormalize, self).__call__(*args, **kwargs)


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

        # Scatter
        self.scatter_artist = self.axes.scatter([], [])
        self.plot_artist = self.axes.plot([], [], 'o', mec='none')[0]
        self.errorbar_artist = self.axes.errorbar([], [], fmt='none')

        # Line
        self.line_artist = self.axes.plot([], [], '-')[0]

        self.mpl_artists = [self.scatter_artist, self.plot_artist,
                            self.errorbar_artist, self.line_artist]
        self.errorbar_index = 2

        self.reset_cache()

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

    def reset_artists(self):

        if self.state.style != 'Scatter':
            offsets = np.zeros((0, 2))
            self.scatter_artist.set_offsets(offsets)
            self.plot_artist.set_data([], [])

        if self.state.style != 'Line':
            self.line_artist.set_data([], [])

    @defer_draw
    def _update_data(self, changed):

        # Layer artist has been cleared already
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

        if self.state.style == 'Scatter':

            if self.state.cmap_mode == 'Fixed' and self.state.size_mode == 'Fixed':
                # In this case we use Matplotlib's plot function because it has much
                # better performance than scatter.
                offsets = np.zeros((0, 2))
                self.scatter_artist.set_offsets(offsets)
                self.plot_artist.set_data(x, y)
            else:
                self.plot_artist.set_data([], [])
                offsets = np.vstack((x, y)).transpose()
                self.scatter_artist.set_offsets(offsets)

            for eartist in list(self.errorbar_artist[2]):
                if eartist is not None:
                    try:
                        eartist.remove()
                    except ValueError:
                        pass
                    except AttributeError:  # Matplotlib < 1.5
                        pass

            if self.state.xerr_visible or self.state.yerr_visible:

                if self.state.xerr_visible and self.state.xerr_att is not None:
                    xerr = self.layer[self.state.xerr_att].ravel()
                else:
                    xerr = None

                if self.state.yerr_visible and self.state.yerr_att is not None:
                    yerr = self.layer[self.state.yerr_att].ravel()
                else:
                    yerr = None

                self.errorbar_artist = self.axes.errorbar(x, y, fmt='none',
                                                          xerr=xerr, yerr=yerr)
                self.mpl_artists[self.errorbar_index] = self.errorbar_artist

        elif self.state.style == 'Line':

            self.line_artist.set_data(x, y)

        else:

            raise NotImplementedError(self.state.style)  # pragma: nocover

    @defer_draw
    def _update_visual_attributes(self, changed, force=False):

        if not self.enabled:
            return

        if self.state.style == 'Scatter':

            if self.state.cmap_mode == 'Fixed' and self.state.size_mode == 'Fixed':

                if force or 'color' in changed:
                    self.plot_artist.set_color(self.state.color)

                if force or 'size' in changed or 'size_scaling' in changed:
                    self.plot_artist.set_markersize(self.state.size *
                                                    self.state.size_scaling)

                artist = self.plot_artist

            else:

                # TEMPORARY: Matplotlib has a bug that causes set_alpha to
                # change the colors back: https://github.com/matplotlib/matplotlib/issues/8953
                if 'alpha' in changed:
                    force = True

                if force or any(prop in changed for prop in CMAP_PROPERTIES):

                    if self.state.cmap_mode == 'Fixed':
                        c = self.state.color
                        vmin = vmax = cmap = None
                    else:
                        c = self.layer[self.state.cmap_att].ravel()
                        vmin = self.state.cmap_vmin
                        vmax = self.state.cmap_vmax
                        cmap = self.state.cmap

                    if self.state.cmap_mode == 'Fixed':
                        self.scatter_artist.set_facecolors(c)
                    else:
                        self.scatter_artist.set_array(c)
                        self.scatter_artist.set_cmap(cmap)
                        if vmin > vmax:
                            self.scatter_artist.set_clim(vmax, vmin)
                            self.scatter_artist.set_norm(InvertedNormalize(vmax, vmin))
                        else:
                            self.scatter_artist.set_clim(vmin, vmax)
                            self.scatter_artist.set_norm(Normalize(vmin, vmax))

                    self.scatter_artist.set_edgecolor('none')

                if force or any(prop in changed for prop in SIZE_PROPERTIES):

                    if self.state.size_mode == 'Fixed':
                        s = self.state.size * self.state.size_scaling
                        s = broadcast_to(s, self.scatter_artist.get_sizes().shape)
                    else:
                        s = self.layer[self.state.size_att].ravel()
                        s = ((s - self.state.size_vmin) /
                             (self.state.size_vmax - self.state.size_vmin)) * 30
                        s *= self.state.size_scaling

                    # Note, we need to square here because for scatter, s is actually
                    # proportional to the marker area, not radius.
                    self.scatter_artist.set_sizes(s ** 2)

                artist = self.scatter_artist

            if self.state.xerr_visible or self.state.yerr_visible:

                for eartist in list(self.errorbar_artist[2]):

                    if eartist is None:
                        continue

                    if force or 'color' in changed:
                        eartist.set_color(self.state.color)

                    if force or 'alpha' in changed:
                        eartist.set_alpha(self.state.alpha)

                    if force or 'visible' in changed:
                        eartist.set_visible(self.state.visible)

                    if force or 'zorder' in changed:
                        eartist.set_zorder(self.state.zorder)

        elif self.state.style == 'Line':

            if force or 'color' in changed:
                self.line_artist.set_color(self.state.color)

            if force or 'linewidth' in changed:
                self.line_artist.set_linewidth(self.state.linewidth)

            if force or 'linestyle' in changed:
                self.line_artist.set_linestyle(self.state.linestyle)

            artist = self.line_artist

        else:

            raise NotImplementedError(self.state.style)  # pragma: nocover

        if force or 'alpha' in changed:
            artist.set_alpha(self.state.alpha)

        if force or 'zorder' in changed:
            artist.set_zorder(self.state.zorder)

        if force or 'visible' in changed:
            artist.set_visible(self.state.visible)

        self.redraw()

    @defer_draw
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

        if force or 'style' in changed:
            self.reset_artists()
            force = True

        if force or len(changed & DATA_PROPERTIES) > 0:
            self._update_data(changed)
            force = True

        if force or len(changed & VISUAL_PROPERTIES) > 0:
            self._update_visual_attributes(changed, force=force)

    @defer_draw
    def update(self):

        self._update_scatter(force=True)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
