from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from mpl_scatter_density import ScatterDensityArtist

from astropy.visualization import ImageNormalize, LinearStretch, SqrtStretch, AsinhStretch, LogStretch

from glue.utils import defer_draw, broadcast_to
from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute

STRETCHES = {'linear': LinearStretch,
             'sqrt': SqrtStretch,
             'arcsinh': AsinhStretch,
             'log': LogStretch}

CMAP_PROPERTIES = set(['cmap_mode', 'cmap_att', 'cmap_vmin', 'cmap_vmax', 'cmap'])
MARKER_PROPERTIES = set(['size_mode', 'size_att', 'size_vmin', 'size_vmax', 'size_scaling', 'size'])
LINE_PROPERTIES = set(['linewidth', 'linestyle'])
DENSITY_PROPERTIES = set(['dpi', 'stretch'])
VISUAL_PROPERTIES = (CMAP_PROPERTIES | MARKER_PROPERTIES | DENSITY_PROPERTIES |
                     LINE_PROPERTIES | set(['color', 'alpha', 'zorder', 'visible']))

DATA_PROPERTIES = set(['layer', 'x_att', 'y_att', 'cmap_mode', 'size_mode',
                       'xerr_att', 'yerr_att', 'xerr_visible', 'yerr_visible',
                       'vector_visible', 'vx_att', 'vy_att', 'vector_arrowhead', 'vector_mode',
                       'vector_origin', 'line_visible', 'markers_visible', 'vector_scaling'])


class InvertedNormalize(Normalize):
    def __call__(self, *args, **kwargs):
        return 1 - super(InvertedNormalize, self).__call__(*args, **kwargs)


def set_mpl_artist_cmap(artist, values, state):
    vmin = state.cmap_vmin
    vmax = state.cmap_vmax
    cmap = state.cmap
    artist.set_array(values)
    artist.set_cmap(cmap)
    if vmin > vmax:
        artist.set_clim(vmax, vmin)
        artist.set_norm(InvertedNormalize(vmax, vmin))
    else:
        artist.set_clim(vmin, vmax)
        artist.set_norm(Normalize(vmin, vmax))


class ScatterLayerArtist(MatplotlibLayerArtist):

    _layer_state_cls = ScatterLayerState

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ScatterLayerArtist, self).__init__(axes, viewer_state,
                                                 layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_scatter)
        self.state.add_global_callback(self._update_scatter)

        # Scatter
        self.scatter_artist = self.axes.scatter([], [])
        self.plot_artist = self.axes.plot([], [], 'o', mec='none')[0]
        self.errorbar_artist = self.axes.errorbar([], [], fmt='none')
        self.vector_artist = None
        self.line_collection = LineCollection(np.zeros((0, 2, 2)))
        self.axes.add_collection(self.line_collection)

        # Scatter density
        self.density_artist = ScatterDensityArtist(self.axes, [], [], color='white')
        self.axes.add_artist(self.density_artist)

        self.mpl_artists = [self.scatter_artist, self.plot_artist,
                            self.errorbar_artist, self.vector_artist,
                            self.line_collection, self.density_artist]
        self.errorbar_index = 2
        self.vector_index = 3

        self.reset_cache()

    def reset_cache(self):
        self._last_viewer_state = {}
        self._last_layer_state = {}

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
            self.enable()

        try:
            y = self.layer[self._viewer_state.y_att].ravel()
        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self._viewer_state.y_att)
            return
        else:
            self.enable()

        if self.state.markers_visible:
            if self.state.cmap_mode == 'Fixed' and self.state.size_mode == 'Fixed':
                # In this case we use Matplotlib's plot function because it has much
                # better performance than scatter.
                self.plot_artist.set_data(x, y)
                offsets = np.zeros((0, 2))
                self.scatter_artist.set_offsets(offsets)
            else:
                self.plot_artist.set_data([], [])
                offsets = np.vstack((x, y)).transpose()
                self.scatter_artist.set_offsets(offsets)
        else:
            self.plot_artist.set_data([], [])
            offsets = np.zeros((0, 2))
            self.scatter_artist.set_offsets(offsets)

        if self.state.line_visible:
            if self.state.cmap_mode == 'Fixed':
                points = np.array([x, y]).transpose()
                self.line_collection.set_segments([points])
            else:
                # In the case where we want to color the line, we need to over
                # sample the line by a factor of two so that we can assign the
                # correct colors to segments - if we didn't do this, then
                # segments on one side of a point would be a different color
                # from the other side. With oversampling, we can have half a
                # segment on either side of a point be the same color as a
                # point
                x_fine = np.zeros(len(x) * 2 - 1, dtype=float)
                y_fine = np.zeros(len(y) * 2 - 1, dtype=float)
                x_fine[::2] = x
                x_fine[1::2] = 0.5 * (x[1:] + x[:-1])
                y_fine[::2] = y
                y_fine[1::2] = 0.5 * (y[1:] + y[:-1])
                points = np.array([x_fine, y_fine]).transpose().reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                self.line_collection.set_segments(segments)
        else:
            self.line_collection.set_segments(np.zeros((0, 2, 2)))

        # TODO: have a visible property
        self.density_artist.set_xy(x, y)

        for eartist in list(self.errorbar_artist[2]):
            if eartist is not None:
                try:
                    eartist.remove()
                except ValueError:
                    pass
                except AttributeError:  # Matplotlib < 1.5
                    pass

        if self.vector_artist is not None:
            self.vector_artist.remove()
            self.vector_artist = None

        if self.state.vector_visible:

            if self.state.vx_att is not None and self.state.vy_att is not None:

                vx = self.layer[self.state.vx_att].ravel()
                vy = self.layer[self.state.vy_att].ravel()

                if self.state.vector_mode == 'Polar':
                    ang = vx
                    length = vy
                    # assume ang is anti clockwise from the x axis
                    vx = length * np.cos(np.radians(ang))
                    vy = length * np.sin(np.radians(ang))

            else:
                vx = None
                vy = None

            if self.state.vector_arrowhead:
                hw = 3
                hl = 5
            else:
                hw = 1
                hl = 0

            v = np.hypot(vx, vy)
            vmax = np.nanmax(v)
            vx = vx / vmax
            vy = vy / vmax

            self.vector_artist = self.axes.quiver(x, y, vx, vy, units='width',
                                                  pivot=self.state.vector_origin,
                                                  headwidth=hw, headlength=hl,
                                                  scale_units='width',
                                                  scale=10 / self.state.vector_scaling)
            self.mpl_artists[self.vector_index] = self.vector_artist

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

    @defer_draw
    def _update_visual_attributes(self, changed, force=False):

        if not self.enabled:
            return

        if self.state.markers_visible:

            if self.state.cmap_mode == 'Fixed' and self.state.size_mode == 'Fixed':

                if force or 'color' in changed:
                    self.plot_artist.set_color(self.state.color)

                if force or 'size' in changed or 'size_scaling' in changed:
                    self.plot_artist.set_markersize(self.state.size *
                                                    self.state.size_scaling)

            else:

                # TEMPORARY: Matplotlib has a bug that causes set_alpha to
                # change the colors back: https://github.com/matplotlib/matplotlib/issues/8953
                if 'alpha' in changed:
                    force = True

                if self.state.cmap_mode == 'Fixed':
                    if force or 'color' in changed or 'cmap_mode' in changed:
                        self.scatter_artist.set_facecolors(self.state.color)
                        self.scatter_artist.set_edgecolor('none')
                elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                    c = self.layer[self.state.cmap_att].ravel()
                    set_mpl_artist_cmap(self.scatter_artist, c, self.state)
                    self.scatter_artist.set_edgecolor('none')

                if force or any(prop in changed for prop in MARKER_PROPERTIES):

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

            # TODO: use this if dealing with the density map

            if force or 'color' in changed:
                self.density_artist.set_color(self.state.color)

            if force or 'stretch' in changed:
                self.density_artist.set_norm(ImageNormalize(stretch=STRETCHES[self.state.stretch]()))

            if force or 'dpi' in changed:
                self.density_artist.set_dpi(self.state.dpi)

        if self.state.line_visible:

            if self.state.cmap_mode == 'Fixed':
                if force or 'color' in changed or 'cmap_mode' in changed:
                    self.line_collection.set_array(None)
                    self.line_collection.set_color(self.state.color)
            elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                # Higher up we oversampled the points in the line so that
                # half a segment on either side of each point has the right
                # color, so we need to also oversample the color here.
                c = self.layer[self.state.cmap_att].ravel()
                cnew = np.zeros((len(c) - 1) * 2)
                cnew[::2] = c[:-1]
                cnew[1::2] = c[1:]
                set_mpl_artist_cmap(self.line_collection, cnew, self.state)

            if force or 'linewidth' in changed:
                self.line_collection.set_linewidth(self.state.linewidth)

            if force or 'linestyle' in changed:
                self.line_collection.set_linestyle(self.state.linestyle)

        if self.state.vector_visible and self.vector_artist is not None:

            if self.state.cmap_mode == 'Fixed':
                if force or 'color' in changed or 'cmap_mode' in changed:
                    self.vector_artist.set_array(None)
                    self.vector_artist.set_color(self.state.color)
            elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                c = self.layer[self.state.cmap_att].ravel()
                set_mpl_artist_cmap(self.vector_artist, c, self.state)

        if self.state.xerr_visible or self.state.yerr_visible:

            for eartist in list(self.errorbar_artist[2]):

                if eartist is None:
                    continue

                if self.state.cmap_mode == 'Fixed':
                    if force or 'color' in changed or 'cmap_mode' in changed:
                        eartist.set_color(self.state.color)
                elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                    c = self.layer[self.state.cmap_att].ravel()
                    set_mpl_artist_cmap(eartist, c, self.state)

                if force or 'alpha' in changed:
                    eartist.set_alpha(self.state.alpha)

                if force or 'visible' in changed:
                    eartist.set_visible(self.state.visible)

                if force or 'zorder' in changed:
                    eartist.set_zorder(self.state.zorder)

        for artist in [self.scatter_artist, self.plot_artist,
                       self.vector_artist, self.line_collection,
                       self.density_artist]:

            if artist is None:
                continue

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

        if force or len(changed & DATA_PROPERTIES) > 0:
            self._update_data(changed)
            force = True

        if force or len(changed & VISUAL_PROPERTIES) > 0:
            self._update_visual_attributes(changed, force=force)

    def get_layer_color(self):
        if self.state.cmap_mode == 'Fixed':
            return self.state.color
        else:
            return self.state.cmap

    @defer_draw
    def update(self):
        self._update_scatter(force=True)
        self.redraw()
