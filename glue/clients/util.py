import logging
from functools import partial
import numpy as np
from matplotlib.ticker import AutoLocator, MaxNLocator, LogLocator
from matplotlib.ticker import (LogFormatterMathtext, ScalarFormatter,
                               FuncFormatter)
from ..core.data import CategoricalComponent
from ..core.decorators import memoize


def get_extent(view, transpose=False):
    sy, sx = [s for s in view if isinstance(s, slice)]
    if transpose:
        return (sy.start, sy.stop, sx.start, sx.stop)
    return (sx.start, sx.stop, sy.start, sy.stop)


def view_cascade(data, view):
    """ Return a set of views progressively zoomed out of input at roughly
    constant pixel count

    :param data: Data object to view
    :param view: Original view into data

    :rtype: tuple of views
    """
    shp = data.shape
    v2 = list(view)
    logging.debug("image shape: %s, view: %s", shp, view)

    # choose stride length that roughly samples entire image
    # at roughly the same pixel count
    step = max(shp[i - 1] * v.step / max(v.stop - v.start, 1)
               for i, v in enumerate(view) if isinstance(v, slice))
    step = max(step, 1)

    for i, v in enumerate(v2):
        if not(isinstance(v, slice)):
            continue
        v2[i] = slice(0, shp[i - 1], step)

    return tuple(v2), view


def small_view(data, attribute):
    """
    Extract a downsampled view from a dataset, for quick
    statistical summaries
    """
    shp = data.shape
    view = tuple([slice(None, None, max(s / 50, 1)) for s in shp])
    return data[attribute, view]


def small_view_array(data):
    """
    Same as small_view, except using a numpy array as input
    """
    shp = data.shape
    view = tuple([slice(None, None, max(s / 50, 1)) for s in shp])
    return np.asarray(data)[view]


@memoize
def fast_limits(data, plo, phi):
    """Quickly estimate percentiles in an array,
    using a downsampled version

    :param data: array-like
    :param plo: Lo percentile
    :param phi: High percentile

    :rtype: Tuple of floats. Approximate values of each percentile in
            data[component]
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("Scale clipping requires SciPy")

    shp = data.shape
    view = tuple([slice(None, None, max(s / 50, 1)) for s in shp])
    values = np.asarray(data)[view]
    if ~np.isfinite(values).any():
        return (0.0, 1.0)

    limits = (-np.inf, np.inf)
    lo = stats.scoreatpercentile(values.flat, plo, limit=limits)
    hi = stats.scoreatpercentile(values.flat, phi, limit=limits)
    return lo, hi


def visible_limits(artists, axis):
    """Determines the data limits for the data in a set of artists

    Ignores non-visible artists

    Assumes each artist as a get_data method wich returns a tuple of x,y

    :param artists: An iterable collection of artists
    :param axis: Which axis to compute. 0=xaxis, 1=yaxis

    :rtype: A tuple of min, max for the requested axis, or None if
            no data present
    """
    data = []
    for art in artists:
        if not art.visible:
            continue
        xy = art.get_data()
        assert isinstance(xy, tuple)
        val = xy[axis]
        if val.size > 0:
            data.append(xy[axis])

    if len(data) == 0:
        return
    data = np.hstack(data)
    if data.size == 0:
        return

    data = data[np.isfinite(data)]
    if data.size == 0:
        return

    lo, hi = np.nanmin(data), np.nanmax(data)
    if not np.isfinite(lo):
        return

    return lo, hi


def tick_linker(all_categories, pos, *args):
    try:
        return all_categories[int(pos)]
    except IndexError:
        return ''


def update_ticks(axes, coord, components, is_log):
    """ Changes the axes to have the proper tick formatting based on the
     type of component.
    :param axes: A matplotlib axis object to alter
    :param coord: 'x' or 'y'
    :param components: A list() of components that are plotted along this axis
    :param is_log: Boolean for log-scale.
    :kwarg max_categories: The maximum number of categories to display.
    :return: None or #categories if components is Categorical
    """

    if coord == 'x':
        axis = axes.xaxis
    elif coord == 'y':
        axis = axes.yaxis
    else:
        raise TypeError("coord must be one of x,y")

    is_cat = all(isinstance(comp, CategoricalComponent) for comp in components)
    if is_log:
        axis.set_major_locator(LogLocator())
        axis.set_major_formatter(LogFormatterMathtext())
    elif is_cat:
        all_categories = np.empty((0,), dtype=np.object)
        for comp in components:
            all_categories = np.union1d(comp._categories, all_categories)
        locator = MaxNLocator(10, integer=True)
        locator.view_limits(0, all_categories.shape[0])
        format_func = partial(tick_linker, all_categories)
        formatter = FuncFormatter(format_func)

        axis.set_major_locator(locator)
        axis.set_major_formatter(formatter)
        return all_categories.shape[0]
    else:
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
