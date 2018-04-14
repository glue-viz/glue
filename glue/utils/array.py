from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.lib.stride_tricks import as_strided

import pandas as pd
import bottleneck as bt

from glue.external.six import string_types
from glue.external.six.moves import range


__all__ = ['unique', 'shape_to_string', 'view_shape', 'stack_view',
           'coerce_numeric', 'check_sorted', 'broadcast_to', 'unbroadcast',
           'iterate_chunks', 'combine_slices', 'nanmean', 'nanmedian', 'nansum',
           'nanmin', 'nanmax', 'format_minimal']


def unbroadcast(array):
    """
    Given an array, return a new array that is the smallest subset of the
    original array that can be re-broadcasted back to the original array.

    See https://stackoverflow.com/questions/40845769/un-broadcasting-numpy-arrays
    for more details.
    """

    if array.ndim == 0:
        return array

    new_shape = np.where(np.array(array.strides) == 0, 1, array.shape)
    return as_strided(array, shape=new_shape)


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

    Returns equivalent of ``np.zeros(shape)[view].shape`` but with minimal
    memory usage.

    Parameters
    ----------
    shape : tuple
        The shape of the array
    view : slice
        A valid index into a Numpy array, or None
    """
    if view is None:
        return shape
    else:
        return np.broadcast_to(1, shape)[view].shape


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

    # Already numeric type
    if np.issubdtype(arr.dtype, np.number):
        return arr

    # Numpy datetime64 format
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr

    # Convert booleans to integers
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int)

    # a string dtype, or anything else
    try:
        return pd.to_numeric(arr, errors='coerce')
    except AttributeError:  # pandas < 0.19
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
        array = np.asarray(array)
        return np.broadcast_arrays(array, np.ones(shape, array.dtype))[0]


def find_chunk_shape(shape, n_max=None):
    """
    Given the shape of an n-dimensional array, and the maximum number of
    elements in a chunk, return the largest chunk shape to use for iteration.

    This currently assumes the optimal chunk shape to return is for C-contiguous
    arrays.
    """

    if n_max is None:
        return tuple(shape)

    block_shape = []

    max_repeat_remaining = n_max

    for size in shape[::-1]:

        if max_repeat_remaining > size:
            block_shape.append(size)
            max_repeat_remaining = max_repeat_remaining // size
        else:
            block_shape.append(max_repeat_remaining)
            max_repeat_remaining = 1

    return tuple(block_shape[::-1])


def iterate_chunks(shape, chunk_shape=None, n_max=None):
    """
    Given a data shape and a chunk shape (or maximum chunk size), iteratively
    return slice objects that can be used to slice the array.
    """

    # Shortcut - if there are any 0 elements in the shape, there are no
    # chunks to iterate over.
    if np.prod(shape) == 0:
        return

    if chunk_shape is None and n_max is None:
        raise ValueError('Either chunk_shape or n_max should be specified')
    elif chunk_shape is not None and n_max is not None:
        raise ValueError('Either chunk_shape or n_max should be specified (not both)')
    elif chunk_shape is None:
        chunk_shape = find_chunk_shape(shape, n_max)
    else:
        if len(chunk_shape) != len(shape):
            raise ValueError('chunk_shape should have the same length as shape')
        elif any(x > y for (x, y) in zip(chunk_shape, shape)):
            raise ValueError('chunk_shape should fit within shape')

    ndim = len(chunk_shape)
    start_index = [0] * ndim

    shape = list(shape)

    while start_index <= shape:

        end_index = [min(start_index[i] + chunk_shape[i], shape[i]) for i in range(ndim)]

        slices = [slice(start_index[i], end_index[i]) for i in range(ndim)]

        yield slices

        # Update chunk index. What we do is to increment the
        # counter for the first dimension, and then if it
        # exceeds the number of elements in that direction,
        # cycle back to zero and advance in the next dimension,
        # and so on.
        start_index[0] += chunk_shape[0]
        for i in range(ndim - 1):
            if start_index[i] >= shape[i]:
                start_index[i] = 0
                start_index[i + 1] += chunk_shape[i + 1]

        # We can now check whether the iteration is finished
        if start_index[-1] >= shape[-1]:
            break


def combine_slices(slice1, slice2, length):
    """
    Given two slices that can be applied to a 1D array and the length of that
    array, this returns a new slice which is the one that should be applied to
    the array instead of slice2 if slice1 has already been applied.
    """

    beg1, end1, step1 = slice1.indices(length)
    beg2, end2, step2 = slice2.indices(length)

    if step1 < 0 or step2 < 0:
        raise ValueError("combine_slices does not support slices with negative step")

    if beg2 >= end1 or end2 <= beg1:
        return slice(0, 0, 1)

    beg = max(beg1, beg2)
    end = min(end1, end2)
    if (beg - beg2) % step2 != 0:
        beg += step2 - ((beg - beg2) % step2)

    # Now we want to find the two first overlap indices inside the overlap
    # range. Loop over indices of second slice (but with min/max constraints
    # of first added) and check if they are valid indices given slice1

    indices = []

    for idx in range(beg, end, step2):
        if (idx - beg1) % step1 == 0:
            indices.append((idx - beg1) // step1)
            if len(indices) == 2:
                break

    if len(indices) == 0:
        return slice(0, 0, 1)
    elif len(indices) == 1:
        return slice(indices[0], indices[0] + 1, 1)
    else:
        end_new = (end - beg1) // step1
        if (end - beg1) % step1 != 0:
            end_new += 1
        return slice(indices[0], end_new, indices[1] - indices[0])


def _move_tuple_axes_first(array, axis):
    """
    Bottleneck can only take integer axis, not tuple, so this function takes all
    the axes to be operated on and combines them into the first dimension of the
    array so that we can then use axis=0
    """

    # Figure out how many axes we are operating over
    naxis = len(axis)

    # Add remaining axes to the axis tuple
    axis += tuple(i for i in range(array.ndim) if i not in axis)

    # The new position of each axis is just in order
    destination = tuple(range(array.ndim))

    # Reorder the array so that the axes being operated on are at the beginning
    array_new = np.moveaxis(array, axis, destination)

    # Figure out the size of the product of the dimensions being operated on
    first = np.prod(array_new.shape[:naxis])

    # Collapse the dimensions being operated on into a single dimension so that
    # we can then use axis=0 with the bottleneck functions
    array_new = array_new.reshape((first,) + array_new.shape[naxis:])

    return array_new


def nanmean(array, axis=None):
    if isinstance(axis, tuple):
        array = _move_tuple_axes_first(array, axis=axis)
        axis = 0
    return bt.nanmean(array, axis=axis)


def nanmedian(array, axis=None):
    if isinstance(axis, tuple):
        array = _move_tuple_axes_first(array, axis=axis)
        axis = 0
    return bt.nanmedian(array, axis=axis)


def nansum(array, axis=None):
    if isinstance(axis, tuple):
        array = _move_tuple_axes_first(array, axis=axis)
        axis = 0
    return bt.nansum(array, axis=axis)


def nanmin(array, axis=None):
    if isinstance(axis, tuple):
        array = _move_tuple_axes_first(array, axis=axis)
        axis = 0
    return bt.nanmin(array, axis=axis)


def nanmax(array, axis=None):
    if isinstance(axis, tuple):
        array = _move_tuple_axes_first(array, axis=axis)
        axis = 0
    return bt.nanmax(array, axis=axis)


def format_minimal(values):
    """
    Find the shortest format that can be used to represent all values in an
    array such that all the string representations are different.

    The current implementation is not incredibly efficient, but it takes only
    ~30ms for a 1000 element array and 200ms for a 10000 element array. One
    could probably make a more efficient implementation but this is good enough
    for now for what we use it for.
    """
    values = np.asarray(values)
    if np.max(np.abs(values)) > 1e5 or np.min(np.diff(values)) < 1e-5:
        fmt_type = 'e'
    else:
        fmt_type = 'f'
    for ndec in range(1, 15):
        fmt = '{{:.{0}{1}}}'.format(ndec, fmt_type)
        strings = [fmt.format(x) for x in values]
        if len(strings) == len(set(strings)):
            break
    return fmt, strings
