import logging
import warnings
from packaging.version import Version
from functools import wraps

import numpy as np
import matplotlib.units as units
import matplotlib.dates as dates
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib import __version__

# We avoid importing matplotlib up here otherwise Matplotlib and therefore Qt
# get imported as soon as glue.utils is imported.

from glue.external.axescache import AxesCache
from glue.utils.misc import DeferredMethod

MATPLOTLIB_GE_30 = Version(__version__) >= Version('3')
MATPLOTLIB_GE_36 = Version(__version__) >= Version('3.6')


__all__ = ['renderless_figure', 'all_artists', 'new_artists', 'remove_artists',
           'get_extent', 'view_cascade', 'fast_limits', 'defer_draw',
           'color2rgb', 'point_contour', 'cache_axes',
           'datetime64_to_mpl', 'mpl_to_datetime64', 'color2hex']


def renderless_figure():
    # Matplotlib figure that skips the render step, for test speed
    from unittest.mock import MagicMock
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.canvas.draw = MagicMock()
    plt.close('all')
    return fig


def all_artists(fig):
    """
    Build a set of all Matplotlib artists in a Figure

    Parameters
    ----------
    fig : :func:`matplotlib.pyplot.figure`
        Matplotlib figure.
    """
    return set(item
               for axes in fig.axes
               for container in [axes.collections, axes.patches, axes.lines,
                                 axes.texts, axes.artists, axes.images]
               for item in container)


def new_artists(fig, old_artists):
    """
    Find the newly-added artists in a figure.

    Parameters
    ----------
    fig : :func:`matplotlib.pyplot.figure`
        Matplotlib figure.
    old_artists : set
        Return value from :func:`all_artists`.

    Returns
    -------
    set
        All artists added since ``all_artists`` was called.
    """
    return all_artists(fig) - old_artists


def remove_artists(artists):
    """
    Remove a collection of matplotlib artists from a scene.

    Parameters
    ----------
    artists : sequence
        Container of artists.
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
    pixel count.

    Parameters
    ----------
    data : array-like
        The array to view.
    view : iterable
        The original view into the data.
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
        if not isinstance(v, slice):
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
    Quickly estimate percentiles in an array, using a downsampled version.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The array to estimate the percentiles for.
    plo, phi : float
        The percentiles.

    Returns
    -------
    lo, hi : float
        The percentile values.
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


# We don't know in advance what backends are going to be used for Matplotlib
# so we can set up a list here and use it in defer_draw below, and each front-
# end is responsible for adding their backend here.
DEFER_DRAW_BACKENDS = []


def defer_draw(func):
    """
    Decorator that globally defers all Agg canvas draws until
    function exit.

    If a Canvas instance's draw method is invoked multiple times,
    it will only be called once after the wrapped function returns.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        if len(DEFER_DRAW_BACKENDS) == 0:
            return func(*args, **kwargs)

        # Don't recursively defer draws. We just check the first draw_idle
        # method since all should be modified in sync.
        if isinstance(DEFER_DRAW_BACKENDS[0].draw_idle, DeferredMethod):
            return func(*args, **kwargs)

        try:
            for backend in DEFER_DRAW_BACKENDS:
                backend.draw_idle = DeferredMethod(backend.draw_idle)
            result = func(*args, **kwargs)
        finally:
            for backend in DEFER_DRAW_BACKENDS:
                # We need to use another try...finally block here in case the
                # executed deferred draw calls fail for any reason
                try:
                    try:
                        backend.draw_idle.execute_deferred_calls()
                    except RuntimeError:  # For C/C++ errors with Qt
                        pass
                finally:
                    backend.draw_idle = backend.draw_idle.original_method

        return result

    wrapper._is_deferred = True
    return wrapper


def color2rgb(color):
    from matplotlib.colors import ColorConverter
    result = ColorConverter().to_rgb(color)
    return result


def color2hex(color):
    try:
        from matplotlib.colors import to_hex
        result = to_hex(color)
    except ImportError:  # MPL 1.5
        from matplotlib.colors import ColorConverter, rgb2hex
        result = rgb2hex(ColorConverter().to_rgb(color))
    return result


def point_contour(x, y, data):
    """Calculate the contour that passes through (x,y) in data

    Parameters
    ----------
    data : `~numpy.ndarray`
        A 2D image.
    x : int
        Index of `x` location.
    y : int
        Index of `y` location.

    Returns
    -------
    contour : `~numpy.ndarray`
        A shape ``(N, 2)`` numpy array giving the `x` and `y` locations
        of the `N` contour vertices.
    """
    try:
        from scipy.ndimage import label, binary_fill_holes
        from skimage.measure import find_contours
    except ImportError:
        raise ImportError("Image processing in Glue requires SciPy and scikit-image")

    # Find the intensity of the selected pixel
    inten = data[y, x]

    # Find all 'islands' above this intensity
    labeled, nr_objects = label(data >= inten)

    # Pick the object we clicked on
    z = (labeled == labeled[y, x])

    # Fill holes inside it so we don't get 'inner' contours
    z = binary_fill_holes(z).astype(float)

    # Pad the resulting array so that for contours that go to the edge we get
    # one continuous contour
    z = np.pad(z, 1, mode='constant')

    # Finally find the contours around the island
    xy = find_contours(z, 0.5, fully_connected='high')

    if not xy:
        return None

    if len(xy) > 1:
        warnings.warn("Too many contours found, picking the first one")

    # We need to flip the array to get (x, y), and subtract one to account for
    # the padding
    return xy[0][:, ::-1] - 1


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
        self.ax.figure.canvas.draw_idle()


def freeze_margins(axes, margins=[1, 1, 1, 1]):
    """
    Make sure margins of axes stay fixed.

    Parameters
    ----------
    ax_class : :class:`matplotlib.axes.Axes`
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
    axes : :class:`matplotlib.axes.Axes`
        The axes to cache
    toolbar : :class:`glue.viewers.common.qt.toolbar.BasicToolbar`
        The toolbar managing the axes' canvas
    """
    canvas = axes.figure.canvas
    cache = AxesCache(axes)
    canvas.resize_begin.connect(cache.enable)
    canvas.resize_end.connect(cache.disable)
    toolbar.pan_begin.connect(cache.enable)
    toolbar.pan_end.connect(cache.disable)
    return cache


class ColormapPatchHandler(HandlerBase):
    def __init__(self, cmap, nb_subpatch=10, xpad=0.0, ypad=0.0):
        """
        A custom legend handler to represent 2D dataset coded in colormaps

        Parameters
        ----------
        cmap : :class:`matplotlib.colors.colormap`
            The matplotlib colormap to use
        nb_subpatch : int, optional
            The number of stripes to use to represent the colormap.
            The default is 10.
        xpad : float, optional
            Padding in the x direction. The default is 0.0.
        ypad : float, optional
            Padding in the y direction. The default is 0.0.

        """
        super().__init__(xpad, ypad)
        self.nb_subpatch = nb_subpatch
        self.cmap = cmap

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        collection = []
        for i in range(self.nb_subpatch):
            width_sub = width / self.nb_subpatch
            x = xdescent + i * width_sub
            collection.append(
                Rectangle((x, ydescent), width_sub, height, transform=trans,
                          facecolor=self.cmap(i / (self.nb_subpatch - 1)),
                          edgecolor="none"))
        return collection

# In Matplotlib < 2.2, there is no datetime64 support, so we register a converter
# here to deal with it with older versions.


class Datetime64Converter(units.ConversionInterface):

    @staticmethod
    def convert(value, unit, axis):
        value = np.asarray(value)
        if value.dtype.kind == 'M':
            return datetime64_to_mpl(value)
        else:
            return value

    @staticmethod
    def axisinfo(unit, axis):
        majloc = dates.AutoDateLocator()
        majfmt = dates.AutoDateFormatter(majloc)
        return units.AxisInfo(majloc=majloc,
                              majfmt=majfmt)

    @staticmethod
    def default_units(x, axis):
        return None


units.registry[np.datetime64] = Datetime64Converter()


# The following code is copied from the developer version of Matplotlib
# for compatibility with older versions. The following is the license
# agreement for Matplotlib:
#
# 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
# ("MDT"), and the Individual or Organization ("Licensee") accessing and
# otherwise using matplotlib software in source or binary form and its
# associated documentation.
#
# 2. Subject to the terms and conditions of this License Agreement, MDT
# hereby grants Licensee a nonexclusive, royalty-free, world-wide license
# to reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use matplotlib
# alone or in any derivative version, provided, however, that MDT's
# License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
# 2012- Matplotlib Development Team; All Rights Reserved" are retained in
# matplotlib  alone or in any derivative version prepared by
# Licensee.
#
# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates matplotlib or any part thereof, and wants to
# make the derivative work available to others as provided herein, then
# Licensee hereby agrees to include in any such work a brief summary of
# the changes made to matplotlib .
#
# 4. MDT is making matplotlib available to Licensee on an "AS
# IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
# IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
# DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
# WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
#
# 5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
#  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
# LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
# MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
# THE POSSIBILITY THEREOF.
#
# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.
#
# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between MDT and
# Licensee.  This License Agreement does not grant permission to use MDT
# trademarks or trade name in a trademark sense to endorse or promote
# products or services of Licensee, or any third party.
#
# 8. By copying, installing or otherwise using matplotlib ,
# Licensee agrees to be bound by the terms and conditions of this License
# Agreement.

HOURS_PER_DAY = 24.
MIN_PER_HOUR = 60.
SEC_PER_MIN = 60.
SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY

T0 = np.datetime64('0001-01-01T00:00:00').astype('datetime64[s]')


def datetime64_to_mpl(d):
    """
    Convert `numpy.datetime64` or an ndarray of those types to Gregorian
    date as UTC float.  The precision is limited to float64 precision. Practically:
    microseconds for dates between 290301 BC, 294241 AD, milliseconds for
    larger dates (see `numpy.datetime64`).  Nanoseconds aren't possible
    because we do times compared to ``0001-01-01T00:00:00`` (plus one day).
    """

    # the "extra" ensures that we at least allow the dynamic range out to
    # seconds.  That should get out to +/-2e11 years.
    extra = d - d.astype('datetime64[s]')
    extra = extra.astype('timedelta64[ns]')
    dt = (d.astype('datetime64[s]') - T0).astype(np.float64)
    dt += extra.astype(np.float64) / 1.0e9
    dt = dt / SEC_PER_DAY + 1.0

    return dt


def mpl_to_datetime64(dt):

    dt = np.asarray(dt, np.float64)

    dt = (dt - 1.0) * SEC_PER_DAY
    dt_s = dt.astype(np.int64) + T0.astype(np.int64)
    dt_ns = ((dt % 1) * 1e9).astype(np.int64)

    dt_s = np.array(dt_s, dtype='datetime64[s]')
    dt_ns = np.array(dt_ns, dtype='timedelta64[ns]')

    return dt_s + dt_ns
