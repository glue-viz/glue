import logging

import numpy as np


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


def file_format(filename):
    if filename.find('.') == -1:
        return ''
    if filename.lower().endswith('.gz'):
        result = filename.lower().rsplit('.', 2)[1]
    else:
        result = filename.lower().rsplit('.', 1)[1]
    return result


def point_contour(x, y, data):
    """Calculate the contour that passes through (x,y) in data

    :param x: x location
    :param y: y location
    :param data: 2D image
    :type data: ndarray

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


def split_component_view(arg):
    """Split the input to data or subset.__getitem__ into its pieces.

    :param arg:
    The input passed to data or subset.__getitem__. Assumed to be either a
    scalar or tuple

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


def view_shape(shape, view):
    """Return the shape of a view of an array

    :param shape: Tuple describing shape of the array
    :param view: View object -- a valid index into a numpy array, or None

    Returns equivalent of np.zeros(shape)[view].shape
    """
    if view is None:
        return shape
    shp = tuple(slice(0, s, 1) for s in shape)
    xy = np.broadcast_arrays(*np.ogrid[shp])
    assert xy[0].shape == shape
    return xy[0][view].shape


def color2rgb(color):
    from matplotlib.colors import ColorConverter
    result = ColorConverter().to_rgb(color)
    return result


def facet_subsets(data, cid, lo=None, hi=None, steps=5,
                  prefix=None, log=False):
    """Create a serries of subsets that partition the values of
    a particular attribute into several bins

    :param data: Data object to use
    :type data: :class:`~glue.core.data.Data`

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

    :param prefix: If present, the new subsets will be labeled `prefix_1`, etc.
    :type prefix: str

    :param log: If True, space divisions logarithmically. Default=False
    :type log: bool

    This creates `steps` new subets, adds them to the data object,
    and returns the list of newly created subsets.

    :rtype: The subsets that were added to `data`

    Example::

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2)

    creates 2 new subsets. The first represents the constraint 0 <=
    mass < 5. The second represents 5 <= mass < 10::

        facet_subset(data, data.id['mass'], lo=10, hi=0, steps=2)

    Creates 2 new subsets. The first represents the constraint 10 >= x > 5
    The second represents 5 >= mass > 0::

        facet_subset(data, data.id['mass'], lo=0, hi=10, steps=2, prefix='m')

    Labels the subsets 'm_1' and 'm_2'

    """
    if lo is None or hi is None:
        vals = data[cid]
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
    for i in range(steps):
        if reverse:
            states.append((cid <= rng[i]) & (cid > rng[i + 1]))

        else:
            states.append((cid >= rng[i]) & (cid < rng[i + 1]))

    result = []
    for i, s in enumerate(states, start=1):
        result.append(data.new_subset())
        result[-1].subset_state = s
        if prefix is not None:
            result[-1].label = "%s_%i" % (prefix, i)

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


def coerce_numeric(arr):
    """Coerce an array into a numeric array, replacing
       non-numeric elements with nans.

       If the array is already a numeric type, it is returned
       unchanged

       :param arr: array to coerce
       :type arr: ndarray instance

       :rtype: ndarray instance.
    """
    # already numeric type
    if np.issubdtype(arr.dtype, np.number):
        return arr

    # a string dtype
    if np.issubdtype(arr.dtype, np.character):
        lens = np.char.str_len(arr)
        lmax = lens.max()
        nonnull = lens > 0
        coerced = np.genfromtxt(arr, delimiter=lmax + 1)
        has_missing = not nonnull.all()
        dtype = np.float if has_missing else coerced.dtype
        result = np.empty(arr.shape, dtype=dtype)
        result[nonnull] = coerced
        if has_missing:
            result[~nonnull] = np.nan
        return result

    return np.genfromtxt(arr)
