from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from glue.external.six import string_types


__all__ = ['unique', 'shape_to_string', 'view_shape', 'stack_view',
           'coerce_numeric', 'check_sorted', 'broadcast_to']


def unique(array):
    """
    Return the unique elements of the array U, as well as
    the index array I such that U[I] == array

    Parameters
    ----------
    array : `numpy.ndarray`
        The array to use

    Returns
    -------
    U : `numpy.ndarray`
        The unique elements of the array
    I : `numpy.ndarray`
        The indices such that ``U[I] == array``
    """
    # numpy.unique doesn't handle mixed-types on python3,
    # so we use pandas
    U, I = pd.factorize(array, sort=True)
    return I, U


def shape_to_string(shape):
    """
    On Windows, shape tuples use long ints which results in formatted shapes
    such as (2L, 3L). This function ensures that the shape is always formatted
    without the Ls.
    """
    return "({0})".format(", ".join(str(int(item)) for item in shape))


def view_shape(shape, view):
    """
    Return the shape of a view of an array.

    Returns equivalent of ``np.zeros(shape)[view].shape``

    Parameters
    ----------
    shape : tuple
        The shape of the array
    view : slice
        A valid index into a Numpy array, or None
    """
    if view is None:
        return shape
    shp = tuple(slice(0, s, 1) for s in shape)
    xy = np.broadcast_arrays(*np.ogrid[shp])
    assert xy[0].shape == shape

    return xy[0][view].shape


def stack_view(shape, *views):
    shp = tuple(slice(0, s, 1) for s in shape)
    result = np.broadcast_arrays(*np.ogrid[shp])
    for v in views:
        if isinstance(v, string_types) and v == 'transpose':
            result = [r.T for r in result]
            continue

        result = [r[v] for r in result]

    return tuple(result)


def coerce_numeric(arr):
    """
    Coerce an array into a numeric array, replacing non-numeric elements with
    nans.

    If the array is already a numeric type, it is returned unchanged

    Parameters
    ----------
    arr : `numpy.ndarray`
        The array to coerce
    """
    # already numeric type
    if np.issubdtype(arr.dtype, np.number):
        return arr

    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int)

    # a string dtype, or anything else
    try:
        return pd.to_numeric(arr, errors='coerce')
    except AttributeError:  # older versions of pandas
        return pd.Series(arr).convert_objects(convert_numeric=True).values


def check_sorted(array):
    """
    Return `True` if the array is sorted, `False` otherwise.
    """
    # this ignores NANs, and does the right thing if nans
    # are concentrated at beginning or end of array
    # otherwise, it will miss things at nan/finite boundaries
    array = np.asarray(array)
    return not (array[:-1] > array[1:]).any()


def pretty_number(numbers):
    """
    Convert a list/array of numbers into a nice list of strings

    Parameters
    ----------
    numbers : list
        The numbers to convert
    """
    try:
        return [pretty_number(n) for n in numbers]
    except TypeError:
        pass

    n = numbers
    if n == 0:
        result = '0'
    elif (abs(n) < 1e-3) or (abs(n) > 1e3):
        result = "%0.3e" % n
    elif abs(int(n) - n) < 1e-3 and int(n) != 0:
        result = "%i" % n
    else:
        result = "%0.3f" % n
        if result.find('.') != -1:
            result = result.rstrip('0')

    return result


def broadcast_to(array, shape):
    """
    Compatibility function - can be removed once we support only Numpy 1.10
    and above
    """
    try:
        return np.broadcast_to(array, shape)
    except AttributeError:
        return array * np.ones(shape, array.dtype)
