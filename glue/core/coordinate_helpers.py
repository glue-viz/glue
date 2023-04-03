import numpy as np
from astropy.wcs import WCS

from glue.utils import unbroadcast
from glue.core.coordinates import LegacyCoordinates


def default_world_coords(wcs):
    if isinstance(wcs, WCS):
        return wcs.wcs.crval
    else:
        return np.zeros(wcs.world_n_dim, dtype=float)


def pixel2world_single_axis(wcs, *pixel, world_axis=None):
    """
    Convert pixel to world coordinates, preserving input type/shape.

    This is a wrapper around pixel_to_world_values which returns the result for
    just one axis, and also determines whether the calculation can be sped up
    if broadcasting is present in the input arrays.

    Parameters
    ----------
    *pixel : array-like
        The pixel coordinates (0-based) to convert.
    world_axis : `int`, optional
        The index of the world coordinate that is needed.

    Returns
    -------
    world : :class:`~numpy.ndarray`
        The world coordinates for the requested axis.
    """

    if world_axis is None:
        raise ValueError("world_axis needs to be set")

    if np.size(pixel[0]) == 0:
        return np.array([], dtype=float)

    original_shape = pixel[0].shape
    pixel_new = []

    # Now find all the pixel coordinates that are needed to calculate this
    # world coordinate, using the axis correlation matrix
    pixel_dep = wcs.axis_correlation_matrix[world_axis, :]

    for ip, p in enumerate(pixel):
        if pixel_dep[ip]:
            pixel_new.append(unbroadcast(p))
        else:
            pixel_new.append(p.flat[0])
    pixel = np.broadcast_arrays(*pixel_new)

    # In the case of 1D WCS, there is an astropy issue which prevents us from
    # passing arbitrary shapes - see https://github.com/astropy/astropy/issues/12154
    # Therefore, we ravel the values and reshape afterwards

    if len(pixel) == 1 and pixel[0].ndim > 1:
        pixel_shape = pixel[0].shape
        result = wcs.pixel_to_world_values(pixel[0].ravel())
        result = result.reshape(pixel_shape)
    else:
        result = wcs.pixel_to_world_values(*pixel)
        if len(pixel) > 1:
            result = result[world_axis]

    return np.broadcast_to(result, original_shape)


def world2pixel_single_axis(wcs, *world, pixel_axis=None):
    """
    Convert world to pixel coordinates, preserving input type/shape.

    This is a wrapper around world_to_pixel_values which returns the result for
    just one axis, and also determines whether the calculation can be sped up
    if broadcasting is present in the input arrays.

    Parameters
    ----------
    *world : array-like
        The world coordinates to convert.
    pixel_axis : `int`, optional
        The index of the pixel coordinate that is needed.

    Returns
    -------
    pixel : :class:`~numpy.ndarray`
        The pixel coordinates for the requested axis.
    """

    if pixel_axis is None:
        raise ValueError("pixel_axis needs to be set")

    if np.size(world[0]) == 0:
        return np.array([], dtype=float)

    original_shape = world[0].shape
    world_new = []

    # Now find all the world coordinates that are needed to calculate this
    # world coordinate, using the axis correlation matrix
    world_dep = wcs.axis_correlation_matrix[:, pixel_axis]

    for iw, w in enumerate(world):
        if world_dep[iw]:
            world_new.append(unbroadcast(w))
        else:
            world_new.append(w.flat[0])
    world = np.broadcast_arrays(*world_new)

    # In the case of 1D WCS, there is an astropy issue which prevents us from
    # passing arbitrary shapes - see https://github.com/astropy/astropy/issues/12154
    # Therefore, we ravel the values and reshape afterwards

    if len(world) == 1 and world[0].ndim > 1:
        world_shape = world[0].shape
        result = wcs.world_to_pixel_values(world[0].ravel())
        result = result.reshape(world_shape)
    else:
        result = wcs.world_to_pixel_values(*world)
        if len(world) > 1:
            result = result[pixel_axis]

    return np.broadcast_to(result, original_shape)


def world_axis(wcs, data, *, pixel_axis=None, world_axis=None):
    """
    Find the world coordinates along a given pixel dimension, and which for now
    we center on the pixel origin.

    Parameters
    ----------
    data : `~glue.core.data.Data`
        The data to compute the coordinate axis for (this is used to
        determine the size of the axis).
    pixel_axis : `int`
        The pixel axis along which to compute the world coordinate,
        in coordinate order.
    world_axis : `int`
        The world axis to compute.

    Notes
    -----
    This method computes the axis values using pixel positions at the
    center of the data along all other axes. This will therefore only give
    the correct result for non-dependent axes (which can be checked using
    the ``dependent_axes`` method).
    """
    pixel = []
    pixel_axis = wcs.pixel_n_dim - 1 - pixel_axis
    for i, s in enumerate(data.shape):
        if i == pixel_axis:
            pixel.append(np.arange(data.shape[pixel_axis]))
        else:
            pixel.append(np.repeat((s - 1) / 2, data.shape[pixel_axis]))
    return pixel2world_single_axis(wcs, *pixel[::-1],
                                        world_axis=world_axis)


def dependent_axes(wcs, axis):
    """
    Return a tuple of which world-axes are non-independent
    from a given pixel axis

    The axis index is given in numpy ordering convention (note that
    opposite the fits convention)
    """
    if isinstance(wcs, LegacyCoordinates):
        return (axis,)
    matrix = wcs.axis_correlation_matrix[::-1, ::-1]
    world_dep = matrix[:, axis:axis + 1]
    return tuple(np.nonzero((world_dep & matrix).any(axis=0))[0])


def _get_ndim(header):
    if 'NAXIS' in header:
        return header['NAXIS']
    if 'WCSAXES' in header:
        return header['WCSAXES']
    return None


def axis_label(wcs, axis):

    if wcs.world_axis_names[axis] != '':
        return wcs.world_axis_names[wcs.world_n_dim - 1 - axis].title()

    if isinstance(wcs, WCS):
        header = wcs.to_header()
        num = _get_ndim(header) - axis  # number orientation reversed
        ax = header.get('CTYPE%i' % num)
        if ax is not None:
            if len(ax) == 8 or '-' in ax:  # assume standard format
                ax = ax[:5].split('-')[0].title()
            else:
                ax = ax.title()

            translate = dict(
                Glon='Galactic Longitude',
                Glat='Galactic Latitude',
                Ra='Right Ascension',
                Dec='Declination',
                Velo='Velocity',
                Freq='Frequency'
            )
            return translate.get(ax, ax)
        unit = header.get('CUNIT%i' % num)
        if unit is not None:
            return "World {} ({})".format(axis, unit)
    return "World {}".format(axis)
