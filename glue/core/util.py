from __future__ import absolute_import, division, print_function

import logging
from contextlib import contextmanager
import string
from itertools import count

import numpy as np
import pandas as pd

from ..external.six.moves import reduce
from ..external.six import string_types


def identity(x):
    return x


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
    """Split the input to data or subset.__getitem__ into its pieces.

    :param arg: The input passed to data or subset.__getitem__.
                Assumed to be either a scalar or tuple

    :rtype: tuple

    The first item is the Component selection (a ComponentID or
    string)

    The second item is a view (tuple of slices, slice scalar, or view
    object)
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
    """Pack a componentID and optional view into single tuple

    Returns an object compatible with data.__getitem__ and related
    methods.  Handles edge cases of when view is None, a scalar, a
    tuple, etc.

    :param component: ComponentID
    :param view: view into data, or None

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
    """Create a series of subsets that partition the values of
    a particular attribute into several bins

    This creates `steps` new subet groups, adds them to the data collection,
    and returns the list of newly created subset groups.

    :param data: DataCollection object to use
    :type data: :class:`~glue.core.data_collection.DataCollection`

    :param cid: ComponentID to facet on
    :type data: :class:`~glue.core.data.ComponentID`

    :param lo: The lower bound for the faceting. Defaults to minimum value
               in data
    :type lo: float

    :param hi: The upper bound for the faceting. Defaults to maximum
               value in data
    :type hi: float

    :param steps: The number of subsets to create. Defaults to 5
    :type steps: int

    :param prefix: If present, the new subset labels will begin with `prefix`
    :type prefix: str

    :param log: If True, space divisions logarithmically. Default=False
    :type log: bool

    :returns: List of :class:`~glue.core.subset_group.SubsetGroup` instances
              added to `data`

    Example::

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2)

    creates 2 new subsets. The first represents the constraint 0 <=
    mass < 5. The second represents 5 <= mass < 10::

        facet_subset(data, data.id['mass'], lo=10, hi=0, steps=2)

    Creates 2 new subsets. The first represents the constraint 10 >= x > 5
    The second represents 5 >= mass > 0::

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2, prefix='m')

    Labels the subsets ``m_1`` and ``m_2``

    """
    from .exceptions import IncompatibleAttribute
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
        if reverse:
            states.append((cid <= rng[i]) & (cid > rng[i + 1]))
            labels.append(prefix + '{0}<{1}<={2}'.format(rng[i + 1], cid, rng[i]))
        else:
            states.append((cid >= rng[i]) & (cid < rng[i + 1]))
            labels.append(prefix + '{0}<={1}<{2}'.format(rng[i], cid, rng[i + 1]))

    result = []
    for lbl, s in zip(labels, states):
        sg = data_collection.new_subset_group(label=lbl, subset_state=s)
        result.append(sg)

    return result


def colorize_subsets(subsets, cmap, lo=0, hi=1):
    """Re-color a list of subsets according to a colormap

    :param subsets: List of subsets
    :param cmap: Matplotlib colormap instance
    :param lo: Start location in colormap. 0-1. Defaults to 0
    :param hi: End location in colormap. 0-1. Defaults to 1

    The colormap will be sampled at `len(subsets)` even intervals
    between `lo` and `hi`. The color at the `ith` interval will be
    applied to `subsets[i]`
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


class PropertySetMixin(object):

    """An object that provides a set of properties that
    are meant to encapsulate state information

    This class exposes a properties attribute, which is a dict
    of all properties. Similarly, assigning to the properties dict
    will update the individual properties
    """
    _property_set = []

    @property
    def properties(self):
        """ A dict mapping property names to values """
        return dict((p, getattr(self, p)) for p in self._property_set)

    @properties.setter
    def properties(self, value):
        """ Update the properties with a new dict.

        Keys in the new dict must be valid property names defined in
        the _property_set class level attribute"""
        invalid = set(value.keys()) - set(self._property_set)
        if invalid:
            raise ValueError("Invalid property values: %s" % invalid)

        for k in self._property_set:
            if k not in value:
                continue
            setattr(self, k, value[k])


class CallbackMixin(object):

    """
    A mixin that provides a utility for attaching callback
    functions to methods
    """

    def __init__(self):
        self._callbacks = []

    def add_callback(self, function):
        self._callbacks.append(function)

    def remove_callback(self, function):
        self._callbacks.remove(function)

    def notify(self, *args, **kwargs):
        for func in self._callbacks:
            func(*args, **kwargs)


class Pointer(object):

    def __init__(self, key):
        self.key = key

    def __get__(self, instance, type=None):
        val = instance
        for k in self.key.split('.'):
            val = getattr(val, k, None)
        return val

    def __set__(self, instance, value):
        v = self.key.split('.')
        attr = reduce(getattr, [instance] + v[:-1])
        setattr(attr, v[-1], value)


@contextmanager
def defer(instance, method):
    """
    Defer the calling of a method inside a context manager,
    and then call it 0 or 1 times afterwards.

    :param instance: The instance of the method to defer
    :param method: The name of the method to defer
    :type method: str

    Within the context block, calls to the method will be
    intercepted, logged, and skipped.

    Upon exiting the context block, the method will be
    invoked a single time, with the arguments of the
    most recent invokation inside the context block.

    If the method is never invoked in the context block,
    it is not called when leaving that block.
    """
    history = []

    def log(*a, **k):
        history.append((a, k))

    orig = getattr(instance, method)
    setattr(instance, method, log)
    try:
        yield
    finally:
        setattr(instance, method, orig)
        for a, k in history[-1:]:
            orig(*a, **k)


def disambiguate(label, taken):
    """If necessary, add a suffix to label to avoid name conflicts

    :param label: desired label
    :param taken: set of taken names

    Returns label if it is not in the taken set. Otherwise, returns
    label_NN where NN is the lowest integer such that label_NN not in taken.
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

    :param data: array-like
    :param categories: array-like of unique values

    :returns: Float array.
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
