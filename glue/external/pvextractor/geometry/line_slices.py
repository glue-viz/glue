from __future__ import print_function

import numpy as np

from scipy.ndimage import map_coordinates


def extract_line_slice(cube, x, y, order=3, respect_nan=True):
    """
    Given an array with shape (z, y, x), extract a (z, n) slice by
    interpolating at n (x, y) points.

    All units are in *pixels*.

    .. note:: If there are NaNs in the cube, they will be treated as zeros when
              using spline interpolation.

    Parameters
    ----------
    cube : `~numpy.ndarray`
        The data cube to extract the slice from
    curve : list or tuple
        A list or tuple of (x, y) pairs, with minimum length 2
    order : int, optional
        Spline interpolation order. Set to ``0`` for nearest-neighbor
        interpolation.

    Returns
    -------
    slice : `numpy.ndarray`
        The (z, d) slice
    """

    if order == 0:

        total_slice = np.zeros([cube.shape[0], len(x)]) + np.nan

        x = np.round(x)
        y = np.round(y)

        ok = (x >= 0) & (y >= 0) & (x < cube.shape[2]) & (y < cube.shape[1])

        total_slice[:,ok] = cube[:, y[ok].astype(int), x[ok].astype(int)]

    elif order > 0 and order == int(order):

        nx = len(x)
        nz = cube.shape[0]

        zi = np.outer(np.arange(nz, dtype=int), np.ones(nx))
        xi = np.outer(np.ones(nz), x)
        yi = np.outer(np.ones(nz), y)

        if np.any(np.isnan(cube)):

            # map_coordinates does not deal well with NaN values so we have
            # to remove the NaN values then re-mask the final slice.

            total_slice = map_coordinates(np.nan_to_num(cube), [zi,yi,xi],
                                          order=order, cval=np.nan)

            slice_bad = map_coordinates(np.nan_to_num(np.isnan(cube).astype(int)),
                                        [zi,yi,xi], order=order)

            total_slice[np.nonzero(slice_bad)] = np.nan

        else:

            total_slice = map_coordinates(cube, [zi,yi,xi], order=order, cval=np.nan)

    else:

        raise TypeError("order should be a positive integer")

    return total_slice
