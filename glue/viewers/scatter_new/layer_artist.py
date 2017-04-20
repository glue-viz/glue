from __future__ import absolute_import, division, print_function

import numpy as np

import matplotlib.pyplot as plt

from glue.utils import nonpartial

from glue.viewers.scatter_new.state import ScatterLayerState
from glue.viewers.common.mpl_layer_artist import MatplotlibLayerArtist

from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from raster_axes.raster_axes import RasterizedScatter, make_colormap
from raster_axes.histogram2d import histogram2d

__all__ = ['ScatterLayerArtist']


class ScatterLayerArtist(MatplotlibLayerArtist):

    def __init__(self, layer, axes, viewer_state, initial_layer_state=None):

        super(ScatterLayerArtist, self).__init__(layer, axes, viewer_state)

        # Set up a state object for the layer artist
        if initial_layer_state is None:
            initial = {}
        else:
            initial = initial_layer_state.as_dict()
            if 'layer' in initial:
                initial.pop('layer')

        self.layer_state = ScatterLayerState(viewer_state=viewer_state, layer=layer, **initial)
        self.viewer_state.layers.append(self.layer_state)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        # TODO: don't connect to ALL signals here
        # self.viewer_state.add_callback('*',nonpartial(self.update))
        self.viewer_state.add_callback('xatt', nonpartial(self.update))
        self.viewer_state.add_callback('yatt', nonpartial(self.update))

        self.layer_state.add_callback('*', nonpartial(self.update))

        # TODO: following is temporary
        self.layer_state.data_collection = self.viewer_state.data_collection
        self.data_collection = self.viewer_state.data_collection

        # Set up an initially empty artists:

        # Scatter
        self.scatter_artist = self.axes.scatter([], [])
        self.plot_artist = self.axes.plot([], [], 'o', mec='none')[0]

        # Fast scatter
        self.norm = ImageNormalize(vmin=0., vmax=1000, stretch=LogStretch())
        self.fast_scatter_artist = RasterizedScatter(self.axes, np.array([]), np.array([]), norm=self.norm)

        # Line
        self.line_artist = self.axes.plot([], [], '-')[0]

        # Histogram
        self.histogram_artist = self.axes.imshow([[]], interpolation='nearest', cmap=plt.cm.gist_heat, aspect='auto', origin='lower')

        # Vectors
        self.vector_artist = self.axes.quiver([], [], [], [])

        self.mpl_artists = [self.scatter_artist, self.plot_artist,
                            self.fast_scatter_artist, self.line_artist,
                            self.histogram_artist, self.vector_artist]

    def reset_artists(self, exception=None):

        if exception != 'Scatter':
            offsets = np.dstack(([], []))
            self.scatter_artist.set_offsets(offsets)
            self.plot_artist.set_data([], [])

        if exception != 'Fast Scatter':
            self.fast_scatter_artist.x = np.array([])
            self.fast_scatter_artist.y = np.array([])
            self.fast_scatter_artist._update(None)

        if exception != 'Line':
            self.line_artist.set_data([], [])

        if exception != '2D Histogram':
            self.histogram_artist.set_data([[]])

        if exception != 'Vectors':
            try:
                self.vector_artist.remove()
            except ValueError:
                pass

    def update(self):

        x = self.layer[self.viewer_state.xatt]
        y = self.layer[self.viewer_state.yatt]

        self.reset_artists(exception=self.layer_state.style)

        if self.layer_state.style == 'Scatter':

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
                    c = self.layer[self.layer_state.cmap_attribute]
                    vmin = self.layer_state.cmap_vmin
                    vmax = self.layer_state.cmap_vmax
                    cmap = self.layer_state.cmap

                if self.layer_state.size_mode == 'Fixed':
                    s = self.layer_state.size
                else:
                    s = self.layer[self.layer_state.size_attribute]
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

        elif self.layer_state.style == 'Fast Scatter':

            self.fast_scatter_artist.x = x.astype(float).ravel()
            self.fast_scatter_artist.y = y.astype(float).ravel()
            self.fast_scatter_artist._update(None)
            self.fast_scatter_artist.set(color=self.layer_state.color, alpha=self.layer_state.alpha)

        elif self.layer_state.style == 'Line':

            self.line_artist.set_data(x.ravel(), y.ravel())
            self.line_artist.set_color(self.layer_state.color)
            self.line_artist.set_zorder(self.zorder)
            self.line_artist.set_alpha(self.layer_state.alpha)
            self.line_artist.set_linewidth(self.layer_state.linewidth)
            self.line_artist.set_linestyle(self.layer_state.linestyle)

        elif self.layer_state.style == '2D Histogram':

            if self.layer_state.h_x_min is None:
                self.layer_state.h_x_min = self.viewer_state.x_min

            if self.layer_state.h_x_max is None:
                self.layer_state.h_x_max = self.viewer_state.x_max

            if self.layer_state.h_y_min is None:
                self.layer_state.h_y_min = self.viewer_state.y_min

            if self.layer_state.h_y_max is None:
                self.layer_state.h_y_max = self.viewer_state.y_max

            array = histogram2d(x.astype(float).ravel(), y.astype(float).ravel(),
                                self.layer_state.h_x_min, self.layer_state.h_x_max,
                                self.layer_state.h_y_min, self.layer_state.h_y_max,
                                self.layer_state.h_nx, self.layer_state.h_ny)

            self.histogram_artist.set_data(array)
            self.histogram_artist.set_extent([self.layer_state.h_x_min, self.layer_state.h_x_max, self.layer_state.h_y_min, self.layer_state.h_y_max])
            self.histogram_artist.set_clim(0, array.max())
            self.histogram_artist.set_cmap(make_colormap(self.layer_state.color))
            self.histogram_artist.set_alpha(self.layer_state.alpha)
            self.histogram_artist.set_zorder(self.zorder)

        elif self.layer_state.style == 'Vectors':

            vx = self.layer[self.layer_state.vector_x_attribute]
            vy = self.layer[self.layer_state.vector_y_attribute]

            U = (vx - self.layer_state.vector_x_min) / (self.layer_state.vector_x_max - self.layer_state.vector_x_min)
            V = (vy - self.layer_state.vector_y_min) / (self.layer_state.vector_y_max - self.layer_state.vector_y_min)

            try:
                self.vector_artist.remove()
            except ValueError:
                pass
            if self.vector_artist in self.mpl_artists:
                self.mpl_artists.remove(self.vector_artist)

            self.vector_artist = self.axes.quiver(x, y, U, V,
                                                  color=self.layer_state.color,
                                                  alpha=self.layer_state.alpha,
                                                  units='dots',
                                                  scale=1. / self.layer_state.vector_scale,
                                                  zorder=self.zorder)

            self.mpl_artists.append(self.vector_artist)

        # Reset the axes stack so that pressing the home button doesn't go back
        # to a previous irrelevant view.
        self.axes.figure.canvas.toolbar.update()

        self.redraw()
