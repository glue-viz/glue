import numpy as np

from .line_slices import extract_line_slice
from .poly_slices import extract_poly_slice


def extract_slice(cube, path, spacing=1.0, order=3, respect_nan=True,
                  wcs=None):
    """
    Given an array with shape (z, y, x), extract a (z, n) slice from a path
    with ``n`` segments.
    
    All units are in *pixels*

    .. note:: If there are NaNs in the cube, they will be treated as zeros when
              using spline interpolation.

    Parameters
    ----------
    path : `Path`
        The path along which to define the slice
    spacing : float
        The position resolution in the final slice
    order : int, optional
        Spline interpolation order when using line paths. Does not have any
        effect for polygon paths.
    respect_nan : bool, optional
        If set to `False`, NaN values are changed to zero before computing
        the slices.

    Returns
    -------
    slice : `numpy.ndarray`
        The slice
    """

    if not respect_nan:
        cube = np.nan_to_num(cube)

    if path.width is None:
        x, y = path.sample_points(spacing=spacing, wcs=wcs)
        slice = extract_line_slice(cube, x, y, order=order)
    else:
        polygons = path.sample_polygons(spacing=spacing, wcs=wcs)
        slice = extract_poly_slice(cube, polygons)

    return slice
