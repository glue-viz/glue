import warnings
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from mpl_scatter_density.generic_density_artist import GenericDensityArtist

from astropy.visualization import (ImageNormalize, LinearStretch, SqrtStretch,
                                   AsinhStretch, LogStretch)

from glue.utils import defer_draw, ensure_numerical, datetime64_to_mpl
from glue.viewers.scatter.state import ScatterLayerState
from glue.viewers.scatter.python_export import python_export_scatter_layer
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute

from matplotlib.lines import Line2D


STRETCHES = {'linear': LinearStretch,
             'sqrt': SqrtStretch,
             'arcsinh': AsinhStretch,
             'log': LogStretch}

CMAP_PROPERTIES = set(['cmap_mode', 'cmap_att', 'cmap_vmin', 'cmap_vmax', 'cmap'])
MARKER_PROPERTIES = set(['size_mode', 'size_att', 'size_vmin', 'size_vmax', 'size_scaling', 'size', 'fill'])
LINE_PROPERTIES = set(['linewidth', 'linestyle'])
DENSITY_PROPERTIES = set(['dpi', 'stretch', 'density_contrast'])
VISUAL_PROPERTIES = (CMAP_PROPERTIES | MARKER_PROPERTIES | DENSITY_PROPERTIES |
                     LINE_PROPERTIES | set(['color', 'alpha', 'zorder', 'visible']))

LIMIT_PROPERTIES = set(['x_min', 'x_max', 'y_min', 'y_max'])
DATA_PROPERTIES = set(['layer', 'x_att', 'y_att', 'cmap_mode', 'size_mode', 'density_map',
                       'xerr_att', 'yerr_att', 'xerr_visible', 'yerr_visible',
                       'vector_visible', 'vx_att', 'vy_att', 'vector_arrowhead', 'vector_mode',
                       'vector_origin', 'line_visible', 'markers_visible', 'vector_scaling'])


def ravel_artists(errorbar_artist):
    for artist_container in errorbar_artist:
        if artist_container is not None:
            for artist in artist_container:
                if artist is not None:
                    yield artist


class InvertedNormalize(Normalize):
    def __call__(self, *args, **kwargs):
        return 1 - super(InvertedNormalize, self).__call__(*args, **kwargs)


class DensityMapLimits(object):

    contrast = 1

    def min(self, array):
        return 0

    def max(self, array):
        return 10. ** (np.log10(np.nanmax(array)) * self.contrast)


def set_mpl_artist_cmap(artist, values, state=None, cmap=None, vmin=None, vmax=None):

    if state is not None:
        vmin = state.cmap_vmin
        vmax = state.cmap_vmax
        cmap = state.cmap
    if not isinstance(artist, GenericDensityArtist):
        artist.set_array(values)
    artist.set_cmap(cmap)
    if vmin > vmax:
        artist.set_clim(vmax, vmin)
        artist.set_norm(InvertedNormalize(vmax, vmin))
    else:
        artist.set_clim(vmin, vmax)
        artist.set_norm(Normalize(vmin, vmax))


class ColoredLineCollection(LineCollection):

    def __init__(self, x, y, **kwargs):
        segments = np.zeros((0, 2, 2))
        super(ColoredLineCollection, self).__init__(segments, **kwargs)
        self.set_points(x, y)

    def set_points(self, x, y, oversample=True):

        if len(x) == 0:
            self.set_segments(np.zeros((0, 2, 2)))
            return

        if oversample:
            x_fine = np.zeros(len(x) * 2 - 1, dtype=float)
            y_fine = np.zeros(len(y) * 2 - 1, dtype=float)
            x_fine[::2] = x
            x_fine[1::2] = 0.5 * (x[1:] + x[:-1])
            y_fine[::2] = y
            y_fine[1::2] = 0.5 * (y[1:] + y[:-1])
            points = np.array([x_fine, y_fine]).transpose().reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            self.set_segments(segments)
        else:
            points = np.array([x, y]).transpose()
            self.set_segments([points])

    def set_linearcolor(self, color=None, data=None, **kwargs):

        if color is None:
            data_new = np.zeros((len(data) - 1) * 2)
            data_new[::2] = data[:-1]
            data_new[1::2] = data[1:]
            self.set_color(None)
            set_mpl_artist_cmap(self, data_new, **kwargs)
        else:
            if isinstance(color, np.ndarray):
                color_new = np.zeros(((color.shape[0] - 1) * 2,) + color.shape[1:])
                color_new[::2] = color[:-1]
                color_new[1::2] = color[1:]
                color = color_new
            self.set_array(None)
            self.set_color(color)


def plot_colored_line(ax, x, y, c=None, cmap=None, vmin=None, vmax=None, **kwargs):
    lc = ColoredLineCollection(x, y, **kwargs)
    lc.set_linearcolor(color=c, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.add_collection(lc)
    return lc


class ScatterLayerArtist(MatplotlibLayerArtist):

    _layer_state_cls = ScatterLayerState
    _python_exporter = python_export_scatter_layer

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ScatterLayerArtist, self).__init__(axes, viewer_state,
                                                 layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_scatter)
        self.state.add_global_callback(self._update_scatter)

        # Scatter density
        self.density_auto_limits = DensityMapLimits()
        self._set_axes(axes)
        self.errorbar_index = 2
        self.vector_index = 3

        # NOTE: Matplotlib can't deal with NaN values in errorbar correctly, so
        # we need to prefilter values - the following variable is used to store
        # the mask for the values we keep, so that we can apply it to the color
        # See also https://github.com/matplotlib/matplotlib/issues/13799
        self._errorbar_keep = None

    def _set_axes(self, axes):
        self.axes = axes
        self.scatter_artist = self.axes.scatter([], [])
        self.plot_artist = self.axes.plot([], [], 'o', mec='none')[0]
        self.errorbar_artist = self.axes.errorbar([], [], fmt='none')
        self.vector_artist = None
        self.line_collection = ColoredLineCollection([], [])
        self.axes.add_collection(self.line_collection)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='All-NaN slice encountered')
            self.density_artist = GenericDensityArtist(self.axes, color='white',
                                                       vmin=self.density_auto_limits.min,
                                                       vmax=self.density_auto_limits.max,
                                                       update_while_panning=False,
                                                       histogram2d_func=self.compute_density_map,
                                                       label=None)
        self.axes.add_artist(self.density_artist)
        self.mpl_artists = [self.scatter_artist, self.plot_artist,
                            self.errorbar_artist, self.vector_artist,
                            self.line_collection, self.density_artist]

    def compute_density_map(self, *args, **kwargs):
        try:
            density_map = self.state.compute_density_map(*args, **kwargs)
        except IncompatibleAttribute:
            self.disable_invalid_attributes(self._viewer_state.x_att,
                                            self._viewer_state.y_att)
            return np.array([[np.nan]])
        else:
            self.enable()
        return density_map

    @defer_draw
    def _update_data(self):

        # Layer artist has been cleared already
        if len(self.mpl_artists) == 0:
            return

        try:
            if not self.state.density_map:
                x = ensure_numerical(self.layer[self._viewer_state.x_att].ravel())
                if x.dtype.kind == 'M':
                    x = datetime64_to_mpl(x)

        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self._viewer_state.x_att)
            return
        else:
            self.enable()

        try:
            if not self.state.density_map:
                y = ensure_numerical(self.layer[self._viewer_state.y_att].ravel())
                if y.dtype.kind == 'M':
                    y = datetime64_to_mpl(y)
        except (IncompatibleAttribute, IndexError):
            # The following includes a call to self.clear()
            self.disable_invalid_attributes(self._viewer_state.y_att)
            return
        else:
            self.enable()

        if self.state.markers_visible:

            if self.state.density_map:
                # We don't use x, y here because we actually make use of the
                # ability of the density artist to call a custom histogram
                # method which is defined on this class and does the data
                # access.
                self.plot_artist.set_data([], [])
                self.scatter_artist.set_offsets(np.zeros((0, 2)))
            else:
                full_sphere = getattr(self._viewer_state, 'using_full_sphere', False)
                degrees = getattr(self._viewer_state, 'using_degrees', False)
                if degrees:
                    x = np.radians(x)
                    if full_sphere:
                        y = np.radians(y)

                # The full-sphere projections expect longitude angles in the range [-pi, pi]
                # so we wrap angles to accommodate this
                if full_sphere:
                    x = np.mod(x + np.pi, 2 * np.pi) - np.pi
                    if self._viewer_state.x_min > self._viewer_state.x_max:
                        x = np.negative(x)
                    if self._viewer_state.y_min > self._viewer_state.y_max:
                        y = np.negative(y)

                self.density_artist.set_label(None)
                if self._use_plot_artist():
                    # In this case we use Matplotlib's plot function because it has much
                    # better performance than scatter.
                    self.plot_artist.set_data(x, y)
                else:
                    offsets = np.vstack((x, y)).transpose()
                    self.scatter_artist.set_offsets(offsets)
        else:
            self.plot_artist.set_data([], [])
            self.scatter_artist.set_offsets(np.zeros((0, 2)))

        if self.state.line_visible:
            if self.state.cmap_mode == 'Fixed':
                self.line_collection.set_points(x, y, oversample=False)
            else:
                # In the case where we want to color the line, we need to over
                # sample the line by a factor of two so that we can assign the
                # correct colors to segments - if we didn't do this, then
                # segments on one side of a point would be a different color
                # from the other side. With oversampling, we can have half a
                # segment on either side of a point be the same color as a
                # point
                self.line_collection.set_points(x, y)
        else:
            self.line_collection.set_points([], [])

        for eartist in ravel_artists(self.errorbar_artist):
            try:
                eartist.remove()
            except ValueError:
                pass

        if self.vector_artist is not None:
            self.vector_artist.remove()
            self.vector_artist = None

        if self.state.vector_visible:

            if self.state.vx_att is not None and self.state.vy_att is not None:

                vx = ensure_numerical(self.layer[self.state.vx_att].ravel())
                vy = ensure_numerical(self.layer[self.state.vy_att].ravel())

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

            vmax = np.nanmax(np.hypot(vx, vy))

            self.vector_artist = self.axes.quiver(x, y, vx, vy, units='width',
                                                  pivot=self.state.vector_origin,
                                                  headwidth=hw, headlength=hl,
                                                  scale_units='width', angles='xy',
                                                  scale=10 / self.state.vector_scaling * vmax
                                                  )
            self.mpl_artists[self.vector_index] = self.vector_artist

        if self.state.xerr_visible or self.state.yerr_visible:

            keep = ~np.isnan(x) & ~np.isnan(y)

            if self.state.xerr_visible and self.state.xerr_att is not None:
                xerr = ensure_numerical(self.layer[self.state.xerr_att].ravel()).copy()
                keep &= ~np.isnan(xerr) & (xerr >= 0.)
            else:
                xerr = None

            if self.state.yerr_visible and self.state.yerr_att is not None:
                yerr = ensure_numerical(self.layer[self.state.yerr_att].ravel()).copy()
                keep &= ~np.isnan(yerr) & (yerr >= 0.)
            else:
                yerr = None

            if xerr is not None:
                xerr = xerr[keep]
            if yerr is not None:
                yerr = yerr[keep]

            self._errorbar_keep = keep

            self.errorbar_artist = self.axes.errorbar(x[keep], y[keep], fmt='none',
                                                      xerr=xerr, yerr=yerr
                                                      )
            self.mpl_artists[self.errorbar_index] = self.errorbar_artist

    @defer_draw
    def _update_visual_attributes(self, changed, force=False):

        if not self.enabled:
            return

        if self.state.markers_visible:

            if self.state.density_map:

                if self.state.cmap_mode == 'Fixed':
                    if force or 'color' in changed or 'cmap_mode' in changed:
                        self.density_artist.set_color(self.state.color)
                        self.density_artist.set_clim(self.density_auto_limits.min,
                                                     self.density_auto_limits.max)
                elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                    c = ensure_numerical(self.layer[self.state.cmap_att].ravel())
                    set_mpl_artist_cmap(self.density_artist, c, self.state)

                if force or 'stretch' in changed:
                    self.density_artist.set_norm(ImageNormalize(stretch=STRETCHES[self.state.stretch]()))

                if force or 'dpi' in changed:
                    self.density_artist.set_dpi(self._viewer_state.dpi)

                if force or 'density_contrast' in changed:
                    self.density_auto_limits.contrast = self.state.density_contrast
                    self.density_artist.stale = True

            else:

                if self._use_plot_artist():

                    if force or 'color' in changed or 'fill' in changed:
                        if self.state.fill:
                            self.plot_artist.set_markeredgecolor('none')
                            self.plot_artist.set_markerfacecolor(self.state.color)
                        else:
                            self.plot_artist.set_markeredgecolor(self.state.color)
                            self.plot_artist.set_markerfacecolor('none')

                    if force or 'size' in changed or 'size_scaling' in changed:
                        self.plot_artist.set_markersize(self.state.size *
                                                        self.state.size_scaling)

                else:

                    # TEMPORARY: Matplotlib has a bug that causes set_alpha to
                    # change the colors back: https://github.com/matplotlib/matplotlib/issues/8953
                    if 'alpha' in changed:
                        force = True

                    if self.state.cmap_mode == 'Fixed':
                        if force or 'color' in changed or 'cmap_mode' in changed or 'fill' in changed:
                            self.scatter_artist.set_array(None)
                            if self.state.fill:
                                self.scatter_artist.set_facecolors(self.state.color)
                                self.scatter_artist.set_edgecolors('none')
                            else:
                                self.scatter_artist.set_facecolors('none')
                                self.scatter_artist.set_edgecolors(self.state.color)
                    elif force or any(prop in changed for prop in CMAP_PROPERTIES) or 'fill' in changed:
                        self.scatter_artist.set_edgecolors(None)
                        self.scatter_artist.set_facecolors(None)
                        c = ensure_numerical(self.layer[self.state.cmap_att].ravel())
                        set_mpl_artist_cmap(self.scatter_artist, c, self.state)
                        if self.state.fill:
                            self.scatter_artist.set_edgecolors('none')
                        else:
                            self.scatter_artist.set_facecolors('none')

                    if force or any(prop in changed for prop in MARKER_PROPERTIES):

                        if self.state.size_mode == 'Fixed':
                            s = self.state.size * self.state.size_scaling
                            s = np.broadcast_to(s, self.scatter_artist.get_sizes().shape)
                        else:
                            s = ensure_numerical(self.layer[self.state.size_att].ravel())
                            s = ((s - self.state.size_vmin) /
                                 (self.state.size_vmax - self.state.size_vmin))
                            # The following ensures that the sizes are in the
                            # range 3 to 30 before the final size_scaling.
                            np.clip(s, 0, 1, out=s)
                            s *= 0.95
                            s += 0.05
                            s *= (30 * self.state.size_scaling)

                        # Note, we need to square here because for scatter, s is actually
                        # proportional to the marker area, not radius.
                        self.scatter_artist.set_sizes(s ** 2)

        if self.state.line_visible:

            if self.state.cmap_mode == 'Fixed':
                if force or 'color' in changed or 'cmap_mode' in changed:
                    self.line_collection.set_linearcolor(color=self.state.color)
            elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                # Higher up we oversampled the points in the line so that
                # half a segment on either side of each point has the right
                # color, so we need to also oversample the color here.
                c = ensure_numerical(self.layer[self.state.cmap_att].ravel())
                self.line_collection.set_linearcolor(data=c, state=self.state)

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
                c = ensure_numerical(self.layer[self.state.cmap_att].ravel())
                set_mpl_artist_cmap(self.vector_artist, c, self.state)

        if self.state.xerr_visible or self.state.yerr_visible:

            for eartist in ravel_artists(self.errorbar_artist):

                if self.state.cmap_mode == 'Fixed':
                    if force or 'color' in changed or 'cmap_mode' in changed:
                        eartist.set_color(self.state.color)
                elif force or any(prop in changed for prop in CMAP_PROPERTIES):
                    c = ensure_numerical(self.layer[self.state.cmap_att].ravel()).copy()
                    c = c[self._errorbar_keep]
                    eartist.set_color(None)
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
                # We need to hide the density artist if it is not needed because
                # otherwise it might still show even if there is no data as the
                # neutral/zero color might not be white.
                if artist is self.density_artist:
                    artist.set_visible(self.state.visible and
                                       self.state.density_map and
                                       self.state.markers_visible)
                else:
                    artist.set_visible(self.state.visible)
        if self._use_plot_artist():
            self.scatter_artist.set_visible(False)
        else:
            self.plot_artist.set_visible(False)
        self.redraw()

    @defer_draw
    def _update_scatter(self, force=False, **kwargs):

        if (self._viewer_state.x_att is None or
            self._viewer_state.y_att is None or
                self.state.layer is None):
            return

        # NOTE: we need to evaluate this even if force=True so that the cache
        # of updated properties is up to date after this method has been called.
        changed = self.pop_changed_properties()

        full_sphere = getattr(self._viewer_state, 'using_full_sphere', False)
        change_from_limits = full_sphere and len(changed & LIMIT_PROPERTIES) > 0
        if force or change_from_limits or len(changed & DATA_PROPERTIES) > 0:
            self._update_data()
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

    def remove(self):
        super(ScatterLayerArtist, self).remove()
        # Clean up the density artist to avoid circular references to do a
        # reference to the self.histogram2d method in density artist.
        self.density_artist = None

    def get_handle_legend(self):
        if self.enabled and self.state.visible:
            handles = []
            if self.state.markers_visible:
                if self.state.density_map:
                    if self.state.cmap_mode == 'Fixed':
                        color = self.get_layer_color()
                    else:
                        color = self.layer.style.color
                    handle = Line2D([0, ], [0, ], marker=".", linestyle="none",
                                    ms=self.state.size, alpha=self.state.alpha,
                                    color=color)
                    handles.append(handle)  # as placeholder
                else:
                    if self._use_plot_artist():
                        handles.append(self.plot_artist)
                    else:
                        handles.append(self.scatter_artist)

            if self.state.line_visible:
                handles.append(self.line_collection)

            if self.state.vector_visible:
                handles.append(self.vector_artist)

            if self.state.xerr_visible or self.state.yerr_visible:
                handles.append(self.errorbar_artist)

            handles = tuple(handles)
            if len(handles) > 0:
                return handles, self.layer.label, None
            else:
                return None, None, None

        else:
            return None, None, None

    def _use_plot_artist(self):
        res = self.state.cmap_mode == 'Fixed' and self.state.size_mode == 'Fixed'
        return res and (not hasattr(self._viewer_state, 'plot_mode') or
                        not self._viewer_state.plot_mode == 'polar')
