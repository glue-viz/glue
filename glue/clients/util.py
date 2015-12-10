from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
from matplotlib.ticker import AutoLocator, MaxNLocator, LogLocator
from matplotlib.ticker import (LogFormatterMathtext, ScalarFormatter,
                               FuncFormatter)


def small_view(data, attribute):
    """
    Extract a downsampled view from a dataset, for quick
    statistical summaries
    """
    shp = data.shape
    view = tuple([slice(None, None, np.int64(max(s / 50, 1))) for s in shp])
    return data[attribute, view]


def small_view_array(data):
    """
    Same as small_view, except using a numpy array as input
    """
    shp = data.shape
    view = tuple([slice(None, None, np.int64(max(s / 50, 1))) for s in shp])
    return np.asarray(data)[view]


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
        pos = np.round(pos)
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

    is_cat = all(comp.categorical for comp in components)
    if is_log:
        axis.set_major_locator(LogLocator())
        axis.set_major_formatter(LogFormatterMathtext())
    elif is_cat:
        all_categories = np.empty((0,), dtype=np.object)
        for comp in components:
            all_categories = np.union1d(comp.categories, all_categories)
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
