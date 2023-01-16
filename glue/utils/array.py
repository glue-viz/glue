import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided

import pandas as pd

# For backward-compatibility with when we used to define bottleneck wrappers
from numpy import nanmin, nanmax, nanmean, nanmedian, nansum  # noqa

__all__ = ['unique', 'shape_to_string', 'view_shape', 'stack_view',
           'coerce_numeric', 'check_sorted', 'unbroadcast',
           'iterate_chunks', 'combine_slices', 'format_minimal', 'compute_statistic',
           'categorical_ndarray', 'index_lookup', 'ensure_numerical',
           'broadcast_arrays_minimal', 'random_views_for_dask_array']


def unbroadcast(array):
    """
    Given an array, return a new array that is the smallest subset of the
    original array that can be re-broadcasted back to the original array.

    See https://stackoverflow.com/questions/40845769/un-broadcasting-numpy-arrays
    for more details.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to use.

    Returns
    -------
    :class:`~numpy.ndarray`
        The reshaped array.
    """

    if array.ndim == 0 or not hasattr(array, 'strides'):
        return array

    new_shape = np.where(np.array(array.strides) == 0, 1, array.shape)
    return as_strided(array, shape=new_shape)


def broadcast_arrays_minimal(*arrays):
    """
    Unbroadcast arrays then broadcast to smallest common shape.

    Parameters
    ----------
    arrays : iterable of `~numpy.ndarray`
        The arrays to broadcast.

    Returns
    -------
    broadcasted : list of `~numpy.ndarray`
        List of views on the unbroadcasted arrays.
    """
    return np.broadcast_arrays(*[unbroadcast(array) for array in arrays])


def unique(array):
    """
    Return the unique elements of the array U, as well as
    the index array I such that U[I] == array

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to use.

    Returns
    -------
    U : `~numpy.ndarray`
        The unique elements of the array
    I : `~numpy.ndarray`
        The indices such that ``U[I] == array``
    """
    # numpy.unique doesn't handle mixed-types on python3,
    # so we use pandas
    array = np.asarray(array)
    I, U = pd.factorize(array.ravel(), sort=True)
    return U.astype(array.dtype), I.reshape(array.shape)


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
        if isinstance(v, str) and v == 'transpose':
            result = [r.T for r in result]
            continue

        result = [r[v] for r in result]

    return tuple(result)


def coerce_numeric(arr):
    """
    Coerce an array into a numeric array, replacing non-numeric elements with
    nans.

    If the array is already a numeric type, it is returned unchanged.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        The array to coerce.
    """

    # Already numeric type
    if np.issubdtype(arr.dtype, np.number):
        return arr

    # Numpy datetime64 format
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr

    # Convert booleans to integers
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(int)

    # a string dtype, or anything else
    try:
        return pd.to_numeric(arr, errors='coerce')
    except AttributeError:  # pandas < 0.19
        return pd.Series(arr).convert_objects(convert_numeric=True).values


def check_sorted(array):
    """
    Return `True` if the array is sorted, `False` otherwise.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to test.
    """
    # this ignores NANs, and does the right thing if nans
    # are concentrated at beginning or end of array
    # otherwise, it will miss things at nan/finite boundaries
    array = np.asarray(array)
    return not (array[:-1] > array[1:]).any()


def pretty_number(numbers):
    """
    Convert an iterable of numbers into a nice list of strings.

    Parameters
    ----------
    numbers : array-like of int or float
        The numbers to convert.
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


def find_chunk_shape(shape, n_max=None):
    """
    Given the shape of an n-dimensional array, and the maximum number of
    elements in a chunk, return the largest chunk shape to use for iteration.

    This currently assumes the optimal chunk shape to return is for C-contiguous
    arrays.

    Parameters
    ----------
    shape : iterable
        The shape of the n-dimensional array.
    n_max : int, optional
        The maximum number of elements per chunk.
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

    Parameters
    ----------
    shape : iterable
        The shape of the n-dimensional array.
    chunk_shape : iterable, optional
        The shape of the chunks to be sliced into, needs to fit within `shape`.
    n_max : int, optional
        The maximum number of elements per chunk.

    Notes
    -----
    Exactly one of `chunk_shape` or `n_max` must be specified.
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

        slices = tuple([slice(start_index[i], end_index[i]) for i in range(ndim)])

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

    Parameters
    ----------
    slice1 : slice
        The first slice (only positive step sizes allowed).
    slice2 : slice
        The second slice (only positive step sizes allowed).
    length : int
        The length of the array on which the combined slice is to be applied.
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


def format_minimal(values):
    """
    Find the shortest `float` format that can be used to represent all values
    in an array such that all the string representations are different.

    The current implementation is not incredibly efficient, but it takes only
    ~30ms for a 1000 element array and 200ms for a 10000 element array. One
    could probably make a more efficient implementation but this is good enough
    for now for what we use it for.

    Returns the optimal format as well as an array of formatted values.

    Parameters
    ----------
    values : array-like
        The (typically non-integer) values to be formatted.
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


def nansum_with_nan_for_empty(values, axis=None):
    """
    Wrapper around nansum which returns NaN instead of zero when no non-NaN
    values are summed.
    """
    result = np.nansum(values, axis=axis)
    reset = np.sum(~np.isnan(values), axis=axis)
    if np.isscalar(result):
        if reset == 0:
            return np.nan
        else:
            return result
    else:
        result[reset == 0] = np.nan
        return result


PLAIN_FUNCTIONS = {'minimum': np.min,
                   'maximum': np.max,
                   'mean': np.mean,
                   'median': np.median,
                   'sum': np.sum,
                   'percentile': np.percentile}

NAN_FUNCTIONS = {'minimum': np.nanmin,
                 'maximum': np.nanmax,
                 'mean': np.nanmean,
                 'median': np.nanmedian,
                 'sum': nansum_with_nan_for_empty,
                 'percentile': np.nanpercentile}


def compute_statistic(statistic, data, mask=None, axis=None, finite=True,
                      positive=False, percentile=None):
    """
    Compute a statistic for the data.

    Parameters
    ----------
    statistic : {'minimum', 'maximum', 'mean', 'median', 'sum', 'percentile'}
        The statistic to compute
    data : `~numpy.ndarray`
        The data to compute the statistic for.
    mask : `~numpy.ndarray`
        The mask to apply when computing the statistic.
    axis : None or int or tuple of int
        If specified, the axis/axes to compute the statistic over.
    finite : bool, optional
        Whether to include only finite values in the statistic. This should
        be `True` to ignore NaN/Inf values
    positive : bool, optional
        Whether to include only (strictly) positive values in the statistic.
        This is used for example when computing statistics of data shown in
        log space.
    percentile : float, optional
        If ``statistic`` is ``'percentile'``, the ``percentile`` argument
        should be given and specify the percentile to calculate in the
        range [0:100]
    """

    data = np.asanyarray(data)
    if mask is not None:
        mask = np.asanyarray(mask, dtype=bool)

    # NOTE: this function should not ever have to use glue-specific objects.
    # The aim is to eventually use a fast C implementation of this function.

    if statistic not in PLAIN_FUNCTIONS:
        raise ValueError("Unrecognized statistic: {0}".format(statistic))

    if (finite or positive or mask is not None) and data.dtype.kind != 'M':

        keep = np.ones(data.shape, dtype=bool)

        if finite:
            keep &= np.isfinite(data)

        if positive:
            keep &= data > 0

        if mask is not None:
            keep &= mask

        if axis is None:
            data = data[keep]
        else:
            # We need to force a copy since we are editing the values and we
            # might as well convert to float just in case
            data = np.array(data, dtype=float)
            data[~keep] = np.nan

        function = NAN_FUNCTIONS[statistic]

    else:

        function = PLAIN_FUNCTIONS[statistic]

    if data.size == 0:
        return np.nan

    if isinstance(axis, tuple) and len(axis) == 0:
        return data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if statistic == 'percentile':
            return function(data, percentile, axis=axis)
        else:
            return function(data, axis=axis)


class categorical_ndarray(np.ndarray):
    """
    A Numpy array subclass that includes properties to find the categories and
    unique integer codes for array values.
    """

    _jitter = None

    def __new__(cls, value, dtype=None, copy=True, order=None, subok=False,
                ndmin=0, categories=None):
        result = np.array(value, dtype=dtype, copy=copy, order=order,
                          subok=True, ndmin=ndmin).view(categorical_ndarray)
        if categories is not None:
            result.categories = categories
        return result

    def __array_finalize__(self, obj):
        if isinstance(obj, categorical_ndarray):
            self.categories = obj.categories

    def _update_categories_and_codes(self):
        if hasattr(self, '_categories'):
            self._codes = index_lookup(self, self._categories)
        else:
            self._categories, self._codes = unique(self)
            self._categories.setflags(write=False)
            self._codes = self._codes.astype(float)
            self._codes.setflags(write=False)

    @property
    def categories(self):
        if not hasattr(self, '_categories'):
            self._update_categories_and_codes()
        return self._categories

    @categories.setter
    def categories(self, value):
        self._categories = value

    @property
    def codes(self):
        if not hasattr(self, '_codes'):
            self._update_categories_and_codes()
        if self._jitter is None:
            return self._codes
        else:
            return self._codes + self._jitter

    def jitter(self, method=None):
        """
        Jitter the codes.

        Parameters
        ----------
        method : {None, 'uniform'}
            If `None`, not jittering is done (or any jittering is undone).
            If ``'uniform'``, the codes are randomized by a uniformly
            distributed random variable.
        """
        if method is None:
            self._jitter = None
        elif method == 'uniform':
            self._jitter = np.random.random(self.shape)
            self._jitter -= 0.5
        else:
            raise ValueError("method should be None or 'uniform'")


def ensure_numerical(values):
    if isinstance(values, categorical_ndarray):
        return values.codes
    else:
        return values


def index_lookup(data, items):
    """
    Lookup which index in items each data value is equal to

    Parameters
    ----------
    data : array-like
        An array-like object that can be used to instantiate a :class:`pandas.DataFrame`.
    items : array-like
        An array-like of unique values in `data`.

    Returns
    -------
    :class:`~numpy.ndarray`
        If result[i] is finite, then data[i] = categories[result[i]]
        Otherwise, data[i] is not in the categories list.
    """

    # np.searchsorted doesn't work on mixed types in Python3

    ndata, ncat = len(data), len(items)
    data = pd.DataFrame({'data': data, 'row': np.arange(ndata)})
    cats = pd.DataFrame({'items': items,
                         'cat_row': np.arange(ncat)})

    m = pd.merge(data, cats, left_on='data', right_on='items')
    result = np.zeros(ndata, dtype=float) * np.nan
    result[np.array(m.row)] = m.cat_row
    return result


def random_views_for_dask_array(array, n_random_samples, n_chunks):
    """
    Return a list of views to extract random values from a dask array in an
    efficient way taking into account the chunk layout. This will return
    n_chunks views such that all views together add up to approximately
    n_random_samples samples.
    """

    # Find the indices of the chunks to extract
    indices = [np.random.randint(dimsize, size=n_chunks) for dimsize in array.numblocks]

    # Determine the boundaries of chunks along each dimension
    chunk_indices = [np.hstack([0, np.cumsum([size for size in sizes])]) for sizes in array.chunks]

    n_per_chunk = n_random_samples // n_chunks

    all_slices = []
    for ichunk in range(n_chunks):
        slices = []
        remaining_size = n_per_chunk
        for idim in range(array.ndim):
            start = chunk_indices[idim][indices[idim][ichunk]]
            stop = chunk_indices[idim][indices[idim][ichunk] + 1]
            if stop - start > remaining_size:
                stop = start + remaining_size
            slices.append(slice(start, stop))
            remaining_size //= (stop - start)
            remaining_size = max(1, remaining_size)
        all_slices.append(tuple(slices))

    return all_slices
