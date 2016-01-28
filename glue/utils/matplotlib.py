from __future__ import absolute_import, division, print_function

import logging
from functools import wraps

import numpy as np

# We avoid importing matplotlib up here otherwise Matplotlib and therefore Qt
# get imported as soon as glue.utils is imported.

from glue.external.axescache import AxesCache
from glue.utils.misc import DeferredMethod


__all__ = ['renderless_figure', 'all_artists', 'new_artists', 'remove_artists',
           'get_extent', 'view_cascade', 'fast_limits', 'defer_draw',
           'color2rgb', 'point_contour', 'cache_axes']


def renderless_figure():
    # Matplotlib figure that skips the render step, for test speed
    from mock import MagicMock
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.canvas.draw = MagicMock()
    plt.close('all')
    return fig


def all_artists(fig):
    """
    Build a set of all Matplotlib artists in a Figure
    """
    return set(item
               for axes in fig.axes
               for container in [axes.collections, axes.patches, axes.lines,
                                 axes.texts, axes.artists, axes.images]
               for item in container)


def new_artists(fig, old_artists):
    """
    Find the newly-added artists in a figure

    :param fig: Matplotlib figure
    :param old_artists: Return value from :func:all_artists
    :returns: All artists added since all_artists was called
    """
    return all_artists(fig) - old_artists


def remove_artists(artists):
    """
    Remove a collection of matplotlib artists from a scene

    :param artists: Container of artists
    """
    for a in artists:
        try:
            a.remove()
        except ValueError:  # already removed
            pass


def get_extent(view, transpose=False):
    sy, sx = [s for s in view if isinstance(s, slice)]
    if transpose:
        return (sy.start, sy.stop, sx.start, sx.stop)
    return (sx.start, sx.stop, sy.start, sy.stop)


def view_cascade(data, view):
    """
    Return a set of views progressively zoomed out of input at roughly constant
    pixel count

    Parameters
    ----------
    data : array-like
        The array to view
    view :
        The original view into the data
    """
    shp = data.shape
    v2 = list(view)
    logging.debug("image shape: %s, view: %s", shp, view)

    # choose stride length that roughly samples entire image
    # at roughly the same pixel count
    step = max(shp[i - 1] * v.step // max(v.stop - v.start, 1)
               for i, v in enumerate(view) if isinstance(v, slice))
    step = max(step, 1)

    for i, v in enumerate(v2):
        if not(isinstance(v, slice)):
            continue
        v2[i] = slice(0, shp[i - 1], step)

    return tuple(v2), view


def _scoreatpercentile(values, percentile, limit=None):
    # Avoid using the scipy version since it is available in Numpy
    if limit is not None:
        values = values[(values >= limit[0]) & (values <= limit[1])]
    return np.percentile(values, percentile)


def fast_limits(data, plo, phi):
    """
    Quickly estimate percentiles in an array, using a downsampled version

    Parameters
    ----------
    data : `numpy.ndarray`
        The array to estimate the percentiles for
    plo, phi : float
        The percentile values

    Returns
    -------
    lo, hi : float
        The percentile values
    """

    shp = data.shape
    view = tuple([slice(None, None, np.intp(max(s / 50, 1))) for s in shp])
    values = np.asarray(data)[view]
    if ~np.isfinite(values).any():
        return (0.0, 1.0)

    limits = (-np.inf, np.inf)
    lo = _scoreatpercentile(values.flat, plo, limit=limits)
    hi = _scoreatpercentile(values.flat, phi, limit=limits)
    return lo, hi


def defer_draw(func):
    """
    Decorator that globally defers all Agg canvas draws until
    function exit.

    If a Canvas instance's draw method is invoked multiple times,
    it will only be called once after the wrapped function returns.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        from matplotlib.backends.backend_agg import FigureCanvasAgg

        # don't recursively defer draws
        if isinstance(FigureCanvasAgg.draw, DeferredMethod):
            return func(*args, **kwargs)

        try:
            FigureCanvasAgg.draw = DeferredMethod(FigureCanvasAgg.draw)
            result = func(*args, **kwargs)
        finally:
            FigureCanvasAgg.draw.execute_deferred_calls()
            FigureCanvasAgg.draw = FigureCanvasAgg.draw.original_method
        return result

    wrapper._is_deferred = True
    return wrapper


def color2rgb(color):
    from matplotlib.colors import ColorConverter
    result = ColorConverter().to_rgb(color)
    return result


def point_contour(x, y, data):
    """Calculate the contour that passes through (x,y) in data

    :param x: x location
    :param y: y location
    :param data: 2D image
    :type data: :class:`numpy.ndarray`

    Returns:

       * A (nrow, 2column) numpy array. The two columns give the x and
         y locations of the contour vertices
    """
    try:
        from scipy import ndimage
    except ImportError:
        raise ImportError("Image processing in Glue requires SciPy")

    inten = data[y, x]
    labeled, nr_objects = ndimage.label(data >= inten)
    z = data * (labeled == labeled[y, x])
    y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    from matplotlib import _cntr
    cnt = _cntr.Cntr(x, y, z)
    xy = cnt.trace(inten)
    if not xy:
        return None
    xy = xy[0]
    return xy


class AxesResizer(object):

    def __init__(self, ax, margins):

        self.ax = ax
        self.margins = margins

    @property
    def margins(self):
        return self._margins

    @margins.setter
    def margins(self, margins):
        self._margins = margins

    def on_resize(self, event):

        fig_width = self.ax.figure.get_figwidth()
        fig_height = self.ax.figure.get_figheight()

        x0 = self.margins[0] / fig_width
        x1 = 1 - self.margins[1] / fig_width
        y0 = self.margins[2] / fig_height
        y1 = 1 - self.margins[3] / fig_height

        dx = max(0.01, x1 - x0)
        dy = max(0.01, y1 - y0)

        self.ax.set_position([x0, y0, dx, dy])
        self.ax.figure.canvas.draw()


def freeze_margins(axes, margins=[1, 1, 1, 1]):
    """
    Make sure margins of axes stay fixed.

    Parameters
    ----------
    ax_class : matplotlib.axes.Axes
        The axes class for which to fix the margins
    margins : iterable
        The margins, in inches. The order of the margins is
        ``[left, right, bottom, top]``

    Notes
    -----
    The object that controls the resizing is stored as the resizer attribute of
    the Axes. This can be used to then change the margins:

        >> ax.resizer.margins = [0.5, 0.5, 0.5, 0.5]

    """

    axes.resizer = AxesResizer(axes, margins)
    axes.figure.canvas.mpl_connect('resize_event', axes.resizer.on_resize)


def cache_axes(axes, toolbar):
    """
    Set up caching for an axes object.

    After this, cached renders will be used to quickly re-render an axes during
    window resizing or interactive pan/zooming.

    This function returns an AxesCache instance.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        The axes to cache
    toolbar : `~glue.viewers.common.qt.toolbar.GlueToolbar`
        The toolbar managing the axes' canvas
    """
    canvas = axes.figure.canvas
    cache = AxesCache(axes)
    canvas.resize_begin.connect(cache.enable)
    canvas.resize_end.connect(cache.disable)
    toolbar.pan_begin.connect(cache.enable)
    toolbar.pan_end.connect(cache.disable)
    return cache
