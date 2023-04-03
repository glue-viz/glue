import logging
from itertools import count
from functools import partial
from fractions import Fraction

import numpy as np

from matplotlib.ticker import AutoLocator, MaxNLocator, LogLocator
from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter, FuncFormatter
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from matplotlib.projections.polar import ThetaFormatter, ThetaLocator
import matplotlib.ticker as mticker

__all__ = ["ThetaRadianFormatter", "ThetaDegreeFormatter", "PolarRadiusFormatter",
           "relim", "split_component_view", "join_component_view",
           "facet_subsets", "colorize_subsets", "disambiguate",
           'small_view', 'small_view_array', 'visible_limits',
           'tick_linker', 'update_ticks']


class ThetaRadianFormatter(mticker.Formatter):
    """
    Used to format the *theta* tick labels in radians
    """

    max_num = 20
    max_den = 20
    limit_den = 10000

    def __init__(self, axis_label=None):
        super().__init__()
        self.axis_label = axis_label

    @classmethod
    def _check_valid(cls, num, den):
        return den != 0 and abs(num) < cls.max_num and den < cls.max_den

    @classmethod
    def _numerator_denominator(cls, x):
        f = Fraction(x / np.pi).limit_denominator(cls.limit_den)
        n, d = f.numerator, f.denominator
        if cls._check_valid(n, d):
            return n, d

        f = Fraction(np.pi / x).limit_denominator(cls.limit_den)
        d, n = f.numerator, f.denominator
        d = d if n >= 0 else -d
        n = abs(n)
        if cls._check_valid(n, d):
            return n, d

        return None, None

    @classmethod
    def rad_fn(cls, x, digits=2):
        n, d = cls._numerator_denominator(x)

        if n is None or d is None:
            return "{value:0.{digits}f}".format(value=x, digits=digits)

        absn = abs(n)
        ns = "" if absn == 1 else str(absn)
        if n == 0:
            return "0"
        elif d == 1:
            return r'${0}\pi$'.format(ns)
        elif abs(n) < cls.max_num and d < cls.max_den:
            sgn = "-" if n < 0 else ""
            return r"${0}{1}\pi/{2}$".format(sgn, ns, d)

    def format_ticks(self, values):
        ticks = super().format_ticks(values)
        if self.axis_label:
            ticks[0] = "{label}={value}".format(label=self.axis_label, value=ticks[0])
        return ticks

    def __call__(self, x, pos=None):
        vmin, vmax = self.axis.get_view_interval()
        d = abs(vmax - vmin)
        digits = max(-int(np.log10(d) - 1.5), 0)
        return ThetaRadianFormatter.rad_fn(x, digits=digits)


class ThetaDegreeFormatter(ThetaFormatter):

    def __init__(self, axis_label=None):
        super().__init__()
        self.axis_label = axis_label

    def format_ticks(self, values):
        ticks = super().format_ticks(values)
        if self.axis_label:
            ticks[0] = "{label}={value}".format(label=self.axis_label, value=ticks[0])
        return ticks


class PolarRadiusFormatter(ScalarFormatter):

    def __init__(self, axis_label=None):
        super().__init__()
        self.axis_label = axis_label

    def format_ticks(self, values):
        ticks = super().format_ticks(values)
        vmin, vmax = self.axis.get_view_interval()

        # Not every tick will necessarily be shown
        # We don't want to include ticks outside the view interval

        # Which direction are the ticks running?
        forwards = vmax > vmin
        if forwards:
            delta, index = -1, -1

            def test(idx):
                return values[idx] > vmax and idx >= -len(values)
        else:
            delta, index = 1, 0

            def test(idx):
                return values[idx] < vmax and idx < len(values)
        while test(index):
            index += delta
        if self.axis_label:
            ticks[index] = "{label}={value}".format(label=self.axis_label, value=ticks[index])
        return ticks


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
    component : :class:`glue.core.component_id.ComponentID`
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
                  prefix='', log=False, style=None, cmap=None):
    """
    Create a series of subsets that partition the values of a particular
    attribute into several bins

    This creates `steps` new subset groups, adds them to the data collection, and
    returns the list of newly created subset groups.

    Parameters
    ----------
    data : :class:`glue.core.data_collection.DataCollection`
        The DataCollection object to use
    cid : :class:`glue.core.component_id.ComponentID`
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
    style : dict, optional
        Any visual attributes specified here will be used when creating subsets
    cmap : :class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap instance used to generate colors.
        If specified, this will override any colors set in `style`

    Returns
    -------
    subset_groups : iterable
        List of :class:`glue.core.subset_group.SubsetGroup` instances added to
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

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2, style=dict(alpha=0.5, markersize=5))

    Performs the same faceting operation in the first example, but now each created subset
    will have an alpha of 0.5 and a marker size of 5.
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
            lo = np.nanmin(vals)
        if hi is None:
            hi = np.nanmax(vals)

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
    style = style or {}
    use_cmap = cmap is not None
    if use_cmap:
        colors = iter(sample_colormap(len(states), cmap))

    for lbl, s in zip(labels, states):
        kwargs = dict(label=lbl, subset_state=s, **style)
        if use_cmap:
            kwargs.update(color=next(colors))
        sg = data_collection.new_subset_group(**kwargs)
        result.append(sg)

    return result


def sample_colormap(nsamples, cmap, lo=0, hi=1):
    """
    Sample a colormap. The colormap will be sampled at `nsamples` even intervals
    between `lo` and `hi`. Results are returned as hexadecimal strings.

     Parameters
    ----------
    nsamples: int
        The number of samples
    cmap : :class:`matplotlib.colors.Colormap`
        Matplotlib colormap instancecmap is not None
    lo : float, optional
        Start location in colormap. 0-1. Defaults to 0
    hi : float, optional
        End location in colormap. 0-1. Defaults to 1
    """

    from matplotlib import cm
    sm = cm.ScalarMappable(cmap=cmap)
    sm.norm.vmin = 0
    sm.norm.vmax = 1

    vals = np.linspace(lo, hi, nsamples)
    rgbas = sm.to_rgba(vals)

    samples = []
    for color in rgbas:
        r, g, b, a = color
        r = int(255 * r)
        g = int(255 * g)
        b = int(255 * b)
        samples.append('#%2.2x%2.2x%2.2x' % (r, g, b))

    return samples


def colorize_subsets(subsets, cmap, lo=0, hi=1):
    """
    Re-color a list of subsets according to a colormap.

    The colormap will be sampled at `len(subsets)` even intervals
    between `lo` and `hi`. The color at the `ith` interval will be
    applied to `subsets[i]`.

    Parameters
    ----------
    subsets : list
        List of subsets
    cmap : :class:`matplotlib.colors.Colormap`
        Matplotlib colormap instance
    lo : float, optional
        Start location in colormap. 0-1. Defaults to 0
    hi : float, optional
        End location in colormap. 0-1. Defaults to 1
    """

    rgbas = sample_colormap(len(subsets), cmap, lo=lo, hi=hi)
    for color, subset in zip(rgbas, subsets):
        subset.style.color = color


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

    Assumes each artist as a get_data method which returns a tuple of x,y

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

    lo, hi = np.nanmin(data), np.nanmax(data)
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


def polar_tick_alignment(value, radians):
    if radians:
        value = value * 180 / np.pi

    if value < 90 or 270 < value:
        return "left"
    elif 90 < value < 270:
        return "right"
    else:
        return "center"


def update_ticks(axes, coord, kinds, is_log, categories, projection='rectilinear', radians=True, label=None):
    """
    Changes the axes to have the proper tick formatting based on the type of
    component.

    Returns `None` or the number of categories if components is Categorical.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        A matplotlib axis object to alter
    coord : { 'x' | 'y' }
        The coordinate axis on which to update the ticks
    components : iterable
        A list of components that are plotted along this axis
    if_log : boolean
        Whether the axis has a log-scale
    projection: str
        The name of the matplotlib projection for the axes object. Defaults to 'rectilinear'.
        Currently only the scatter viewer supports different projections.
    """

    if projection == 'lambert' and coord == 'y':
        return

    if coord == 'x':
        axis = axes.xaxis
    elif coord == 'y':
        axis = axes.yaxis
    else:
        raise TypeError("coord must be one of x,y")

    is_cat = 'categorical' in kinds
    is_date = 'datetime' in kinds

    if is_date:
        loc = AutoDateLocator()
        fmt = AutoDateFormatter(loc)
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
    elif is_log:
        axis.set_major_locator(LogLocator())
        axis.set_major_formatter(LogFormatterMathtext())
    elif is_cat:
        locator = MaxNLocator(10, integer=True)
        locator.view_limits(0, categories.shape[0])
        format_func = partial(tick_linker, categories)
        formatter = FuncFormatter(format_func)
        axis.set_major_locator(locator)
        axis.set_major_formatter(formatter)
    # Have to treat the axes for polar plots differently
    elif projection == 'polar':
        if coord == 'x':
            axis.set_major_locator(ThetaLocator(AutoLocator()))
            formatter_type = ThetaRadianFormatter if radians else ThetaDegreeFormatter
            axis.set_major_formatter(formatter_type(label))
            for lbl, loc in zip(axis.get_majorticklabels(), axis.get_majorticklocs()):
                lbl.set_horizontalalignment(polar_tick_alignment(loc, radians))
        else:
            axis.set_major_locator(AutoLocator())
            axis.set_major_formatter(PolarRadiusFormatter(label))
            for lbl in axis.get_majorticklabels():
                lbl.set_fontstyle("italic")
    elif projection in ['aitoff', 'hammer', 'mollweide', 'lambert']:
        formatter_type = ThetaRadianFormatter if radians else ThetaDegreeFormatter
        axis.set_major_formatter(formatter_type())
    else:
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
