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
        return x * .98, y * 1.02
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
