"""
Classes to perform aggregations over cubes
"""

import numpy as np


def mom1(data, axis=0):
    """
    Intensity-weighted coordinate (function version). Pixel units.
    """

    shp = list(data.shape)
    n = shp.pop(axis)

    result = np.zeros(shp)
    w = np.zeros(shp)

    # build up slice-by-slice, to avoid big temporary cubes
    for loc in range(n):
        slc = tuple(loc if j == axis else slice(None) for j in range(data.ndim))
        val = data[slc]
        val = np.maximum(val, 0)
        result += val * loc
        w += val
    return result / w


def mom2(data, axis=0):
    """
    Intensity-weighted coordinate dispersion (function version). Pixel units.
    """

    shp = list(data.shape)
    n = shp.pop(axis)

    x = np.zeros(shp)
    x2 = np.zeros(shp)
    w = np.zeros(shp)

    # build up slice-by-slice, to avoid big temporary cubes
    for loc in range(n):
        slc = tuple(loc if j == axis else slice(None) for j in range(data.ndim))
        val = data[slc]
        val = np.maximum(val, 0)
        x += val * loc
        x2 += val * loc * loc
        w += val
    return np.sqrt(x2 / w - (x / w) ** 2)
