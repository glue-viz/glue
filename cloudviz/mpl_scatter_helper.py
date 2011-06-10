import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def make_colormap(color):

    r, g, b = colors.colorConverter.to_rgb(color)

    cdict = {'red': [(0.0, r, r),
                    (1.0, 1.0, 1.0)],

             'green': [(0.0, g, g),
                       (1.0, 1.0, 1.0)],

             'blue':  [(0.0, b, b),
                       (1.0, 1.0, 1.0)]}

    return colors.LinearSegmentedColormap('custom', cdict)


def warn_not_implemented(kwargs):
    for kwarg in kwargs:
        print "WARNING: keyword argument %s not implemented in raster scatter" % kwarg


class Scatter(object):

    def __init__(self, ax, x, y, color='black', alpha=1.0, **kwargs):

        warn_not_implemented(kwargs)

        self._ax = ax
        self._ax.callbacks.connect('ylim_changed', self._update)

        self.x = x
        self.y = y

        self._color = color
        self._alpha = alpha

        self._raster = None

        self._update(None)

    def set_visible(self, visible):
        self._raster.set_visible(visible)

    def set_offsets(self, coords):
        self.x, self.y = zip(*coords)
        self._update(None)

    def set(self, color=None, alpha=None, **kwargs):

        warn_not_implemented(kwargs)

        if color is not None:
            self._color = color
            self._raster.set_cmap(make_colormap(self._color))

        if alpha is not None:
            self._alpha = alpha
            self._raster.set_alpha(self._alpha)

        if self._color == 'red':
            self._raster.set_zorder(20)


        self._ax.figure.canvas.draw()

    def _update(self, event):

        dpi = self._ax.figure.get_dpi()

        autoscale = self._ax.get_autoscale_on()

        if autoscale:
            self._ax.set_autoscale_on(False)

        width = self._ax.get_position().width \
                * self._ax.figure.get_figwidth()
        height = self._ax.get_position().height \
                 * self._ax.figure.get_figheight()

        nx = int(round(width * dpi))
        ny = int(round(height * dpi))

        xmin, xmax = self._ax.get_xlim()
        ymin, ymax = self._ax.get_ylim()

        ix = ((self.x - xmin) / (xmax - xmin) * float(nx)).astype(int)
        iy = ((self.y - ymin) / (ymax - ymin) * float(ny)).astype(int)

        array = ma.zeros((nx, ny), dtype=int)

        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        array[ix[keep], iy[keep]] += 1

        array = array.transpose()

        array.mask = array == 0

        if self._raster is None:
            self._raster = self._ax.imshow(array,
                                           extent=[xmin, xmax, ymin, ymax],
                                           aspect='auto',
                                           cmap=make_colormap(self._color),
                                           interpolation='nearest',
                                           alpha=self._alpha, origin='lower', zorder=10)
        else:
            self._raster.set_data(array)
            self._raster.set_extent([xmin, xmax, ymin, ymax])

        if autoscale:
            self._ax.set_autoscale_on(True)


class RasterAxes(plt.Axes):

    def __init__(self, *args, **kwargs):
        plt.Axes.__init__(self, *args, **kwargs)
        self._scatter_objects = {}

    def scatter(self, x, y, color='black', alpha=1.0, **kwargs):
        self.set_xlim(np.min(x), np.max(x))
        self.set_ylim(np.min(y), np.max(y))
        scatter = Scatter(self, x, y, color=color, alpha=alpha, **kwargs)
        self._scatter_objects[id(x)] = scatter
        return scatter
