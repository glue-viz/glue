from __future__ import print_function

import numpy as np

from astropy import units as u
from astropy.extern import six
from astropy.io.fits import PrimaryHDU, ImageHDU, Header

from .utils.wcs_utils import get_spatial_scale, sanitize_wcs
from .geometry import extract_slice
from .geometry import path as paths
from .utils.wcs_slicing import slice_wcs


def extract_pv_slice(cube, path, wcs=None, spacing=1.0, order=3,
                     respect_nan=True):
    """
    Given a position-position-velocity cube with dimensions (nv, ny, nx), and
    a path, extract a position-velocity slice.

    Alternative implementations:
        gipsy::sliceview
        karma::kpvslice
        casaviewer::slice

    Parameters
    ----------
    cube : :class:`~numpy.ndarray` or :class:`~spectral_cube.SpectralCube` or str or HDU
        The cube to extract a slice from. If this is a plain
        :class:`~numpy.ndarray` instance, the WCS information can optionally
        be specified with the ``wcs`` parameter. If a string, it should be
        the name of a file containing a spectral cube.
    path : `Path` or list of 2-tuples
        The path along which to define the position-velocity slice. The path
        can contain coordinates defined in pixel or world coordinates.
    wcs : :class:`~astropy.wcs.WCS`, optional
        The WCS information to use for the cube. This should only be
        specified if the ``cube`` parameter is a plain
        :class:`~numpy.ndarray` instance.
    spacing : float
        The position resolution in the final position-velocity slice. This
        can be given in pixel coordinates or as a
        :class:`~astropy.units.Quantity` instance with angle units.
    order : int, optional
        Spline interpolation order when using paths with zero width. Does not
        have any effect for paths with a non-zero width.
    respect_nan : bool, optional
        If set to `False`, NaN values are changed to zero before computing
        the slices. If set to `True`, in the case of line paths a second
        computation is performed to ignore the NaN value while interpolating,
        and set the output values of NaNs to NaN.

    Returns
    -------
    slice : `PrimaryHDU`
        The position-velocity slice, as a FITS HDU object
    """

    if isinstance(cube, (six.string_types, ImageHDU, PrimaryHDU)):
        try:
            from spectral_cube import SpectralCube
            cube = SpectralCube.read(cube)
        except ImportError:
            raise ImportError("spectral_cube package required for working "
                              "with fits data. Install spectral_cube or "
                              "use NumPy arrays")

    if _is_spectral_cube(cube):
        wcs = cube.wcs
        # The fits HEADER will preserve the UNIT, but pvextractor does not care
        # what the flux units are
        cube = cube.filled_data[...].value

    if wcs is not None:
        wcs = sanitize_wcs(wcs)

    if not isinstance(cube, np.ndarray) or wcs is not None:
        scale = get_spatial_scale(wcs)
        if isinstance(spacing, u.Quantity):
            pixel_spacing = (spacing / scale).decompose()
            world_spacing = spacing
        else:
            pixel_spacing = spacing
            world_spacing = spacing * scale
    else:
        if isinstance(spacing, u.Quantity):
            raise TypeError("No WCS has been specified, so spacing should be given in pixels")
        else:
            pixel_spacing = spacing
            world_spacing = None

    # Allow path to be passed in as list of 2-tuples
    if not isinstance(path, paths.Path):
        path = paths.Path(path)

    pv_slice = extract_slice(cube, path, wcs=wcs, spacing=pixel_spacing,
                             order=order, respect_nan=respect_nan)

    # Generate output header
    if wcs is None:
        header = Header()
    else:
        header = slice_wcs(wcs, spatial_scale=world_spacing).to_header()

    # TODO: write path to BinTableHDU

    return PrimaryHDU(data=pv_slice, header=header)


def _is_spectral_cube(obj):
    try:
        from spectral_cube import SpectralCube
        return isinstance(obj, SpectralCube)
    except ImportError:
        return False
