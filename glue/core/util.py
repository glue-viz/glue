from __future__ import absolute_import, division, print_function

import logging
from itertools import count
from functools import partial


import numpy as np
import pandas as pd

from matplotlib.ticker import AutoLocator, MaxNLocator, LogLocator
from matplotlib.ticker import (LogFormatterMathtext, ScalarFormatter,
                               FuncFormatter)
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

from glue.utils import nanmin, nanmax

__all__ = ["relim", "split_component_view", "join_component_view",
           "facet_subsets", "colorize_subsets", "disambiguate",
           "row_lookup", 'small_view', 'small_view_array', 'visible_limits',
           'tick_linker', 'update_ticks']


def relim(lo, hi, log=False):
    logging.getLogger(__name__).debug("Inputs to relim: %r %r", lo, hi)
    x, y = lo, hi
    if log:
        if lo < 0:
            x = 1e-5
        if hi < 0:
            y = 1e5
        return x * .95, y * 1.05
    delta = y - x
    return (x - .02 * delta, y + .02 * delta)


def split_component_view(arg):
    """
    Split the input to data or subset.__getitem__ into its pieces.

    Parameters
    ----------
    arg
        The input passed to ``data`` or ``subset.__getitem__``. Assumed to be
        either a scalar or tuple

    Returns
    -------
    selection
        The Component selection (a ComponentID or string)
    view
        Tuple of slices, slice scalar, or view
    """
    if isinstance(arg, tuple):
        if len(arg) == 1:
            raise TypeError("Expected a scalar or >length-1 tuple, "
                            "got length-1 tuple")
        if len(arg) == 2:
            return arg[0], arg[1]
        return arg[0], arg[1:]
    else:
        return arg, None


def join_component_view(component, view):
    """
    Pack a ComponentID and optional view into single tuple.

    Returns an object compatible with ``data.__getitem__`` and related methods.
    Handles edge cases of when view is None, a scalar, a tuple, etc.

    Parameters
    ----------
    component : `~glue.core.component_id.ComponentID`
        The ComponentID to pack
    view
        The view into the data, or `None`

    """
    if view is None:
        return component
    result = [component]
    try:
        result.extend(view)
    except TypeError:  # view is a scalar
        result = [component, view]

    return tuple(result)


def facet_subsets(data_collection, cid, lo=None, hi=None, steps=5,
                  prefix='', log=False):
    """
    Create a series of subsets that partition the values of a particular
    attribute into several bins

    This creates `steps` new subet groups, adds them to the data collection, and
    returns the list of newly created subset groups.

    Parameters
    ----------
    data : :class:`~glue.core.data_collection.DataCollection`
        The DataCollection object to use
    cid : :class:`~glue.core.component_id.ComponentID`
         The ComponentID to facet on
    lo : float, optional
        The lower bound for the faceting. Defaults to minimum value in data
    hi : float, optional
        The upper bound for the faceting. Defaults to maximum value in data
    steps : int, optional
        The number of subsets to create. Defaults to 5
    prefix : str, optional
        If present, the new subset labels will begin with `prefix`
    log : bool, optional
        If `True`, space divisions logarithmically. Default is `False`

    Returns
    -------
    subset_groups : iterable
        List of :class:`~glue.core.subset_group.SubsetGroup` instances added to
        `data`

    Examples
    --------

    ::

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2)

    creates 2 new subsets. The first represents the constraint 0 <=
    mass < 5. The second represents 5 <= mass <= 10::

        facet_subset(data, data.id['mass'], lo=10, hi=0, steps=2)

    Creates 2 new subsets. The first represents the constraint 10 >= x > 5
    The second represents 5 >= mass >= 0::

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2, prefix='m')

    Labels the subsets ``m_1`` and ``m_2``.

    Note that the last range is inclusive on both sides. For example, if ``lo``
    is 0 and ``hi`` is 5, and ``steps`` is 5, then the intervals for the subsets
    are [0,1), [1,2), [2,3), [3,4), and [4,5].
    """
    from glue.core.exceptions import IncompatibleAttribute
    if lo is None or hi is None:
        for data in data_collection:
            try:
                vals = data[cid]
                break
            except IncompatibleAttribute:
                continue
        else:
            raise ValueError("Cannot infer data limits for ComponentID %s"
                             % cid)
        if lo is None:
            lo = nanmin(vals)
        if hi is None:
            hi = nanmax(vals)

    reverse = lo > hi
    if log:
        rng = np.logspace(np.log10(lo), np.log10(hi), steps + 1)
    else:
        rng = np.linspace(lo, hi, steps + 1)

    states = []
    labels = []
    for i in range(steps):

        # The if i < steps - 1 clauses are needed because the last interval
        # has to be inclusive on both sides.

        if reverse:
            if i < steps - 1:
                states.append((cid <= rng[i]) & (cid > rng[i + 1]))
                labels.append(prefix + '{0}<{1}<={2}'.format(rng[i + 1], cid, rng[i]))
            else:
                states.append((cid <= rng[i]) & (cid >= rng[i + 1]))
                labels.append(prefix + '{0}<={1}<={2}'.format(rng[i + 1], cid, rng[i]))

        else:
            if i < steps - 1:
                states.append((cid >= rng[i]) & (cid < rng[i + 1]))
                labels.append(prefix + '{0}<={1}<{2}'.format(rng[i], cid, rng[i + 1]))
            else:
                states.append((cid >= rng[i]) & (cid <= rng[i + 1]))
                labels.append(prefix + '{0}<={1}<={2}'.format(rng[i], cid, rng[i + 1]))

    result = []
    for lbl, s in zip(labels, states):
        sg = data_collection.new_subset_group(label=lbl, subset_state=s)
        result.append(sg)

    return result


def colorize_subsets(subsets, cmap, lo=0, hi=1):
    """
    Re-color a list of subsets according to a colormap.

    The colormap will be sampled at `len(subsets)` even intervals
    between `lo` and `hi`. The color at the `ith` interval will be
    applied to `subsets[i]`

    Parameters
    ----------
    subsets : list
        List of subsets
    cmap : `~matplotlib.colors.Colormap`
        Matplotlib colormap instance
    lo : float, optional
        Start location in colormap. 0-1. Defaults to 0
    hi : float, optional
        End location in colormap. 0-1. Defaults to 1
    """

    from matplotlib import cm
    sm = cm.ScalarMappable(cmap=cmap)
    sm.norm.vmin = 0
    sm.norm.vmax = 1

    vals = np.linspace(lo, hi, len(subsets))
    rgbas = sm.to_rgba(vals)

    for color, subset in zip(rgbas, subsets):
        r, g, b, a = color
        r = int(255 * r)
        g = int(255 * g)
        b = int(255 * b)
        subset.style.color = '#%2.2x%2.2x%2.2x' % (r, g, b)


def disambiguate(label, taken):
    """
    If necessary, add a suffix to label to avoid name conflicts

    Returns label if it is not in the taken set. Otherwise, returns label_NN
    where NN is the lowest integer such that label_NN not in taken.

    Parameters
    ----------
    label : str
        Desired label
    taken : iterable
        The set of already taken names
    """
    if label not in taken:
        return label
    suffix = "_%2.2i"
    label = str(label)
    for i in count(1):
        candidate = label + (suffix % i)
        if candidate not in taken:
            return candidate


def row_lookup(data, categories):
    """
    Lookup which row in categories each data item is equal to

    Parameters
    ----------
    data
        An array-like object
    categories
        Array-like of unique values

    Returns
    -------
    array
      If result[i] is finite, then data[i] = categoreis[result[i]]
      Otherwise, data[i] is not in the categories list
    """

    # np.searchsorted doesn't work on mixed types in Python3

    ndata, ncat = len(data), len(categories)
    data = pd.DataFrame({'data': data, 'row': np.arange(ndata)})
    cats = pd.DataFrame({'categories': categories,
                         'cat_row': np.arange(ncat)})

    m = pd.merge(data, cats, left_on='data', right_on='categories')
    result = np.zeros(ndata, dtype=float) * np.nan
    result[np.array(m.row)] = m.cat_row
    return result


def small_view(data, attribute):
    """
    Extract a downsampled view from a dataset, for quick statistical summaries
    """
    shp = data.shape
    view = tuple([slice(None, None, np.intp(max(s / 50, 1))) for s in shp])
    return data[attribute, view]


def small_view_array(data):
    """
    Same as small_view, except using a numpy array as input
    """
    shp = data.shape
    view = tuple([slice(None, None, np.intp(max(s / 50, 1))) for s in shp])
    return np.asarray(data)[view]


def visible_limits(artists, axis):
    """
    Determines the data limits for the data in a set of artists.

    Ignores non-visible artists

    Assumes each artist as a get_data method wich returns a tuple of x,y

    Returns a tuple of min, max for the requested axis, or None if no data
    present

    Parameters
    ----------
    artists : iterable
        An iterable collection of artists
    axis : int
        Which axis to compute. 0=xaxis, 1=yaxis
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

    lo, hi = nanmin(data), nanmax(data)
    if not np.isfinite(lo):
        return

    return lo, hi


def tick_linker(all_categories, pos, *args):

    # We need to take care to ignore negative indices since these would actually
    # 'work' 'when accessing all_categories, but we need to avoid that.
    if pos < 0 or pos >= len(all_categories):
        return ''
    else:
        try:
            pos = np.round(pos)
            label = all_categories[int(pos)]
            if isinstance(label, bytes):
                return label.decode('ascii')
            else:
                return label
        except IndexError:
            return ''


def update_ticks(axes, coord, components, is_log):
    """
    Changes the axes to have the proper tick formatting based on the type of
    component.

    Returns `None` or the number of categories if components is Categorical.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        A matplotlib axis object to alter
    coord : { 'x' | 'y' }
        The coordinate axis on which to update the ticks
    components : iterable
        A list of components that are plotted along this axis
    if_log : boolean
        Whether the axis has a log-scale
    """

    if coord == 'x':
        axis = axes.xaxis
    elif coord == 'y':
        axis = axes.yaxis
    else:
        raise TypeError("coord must be one of x,y")

    is_cat = any(comp.categorical for comp in components)
    is_date = any(comp.datetime for comp in components)

    if is_date:
        loc = AutoDateLocator()
        fmt = AutoDateFormatter(loc)
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
    elif is_log:
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
