import numpy as np

from astropy import units as u
from astropy.wcs import WCS, WCSSUB_SPECTRAL
from .wcs_utils import get_spectral_scale


def slice_wcs(wcs, spatial_scale):
    """
    Slice a WCS header for a spectral cube to a Position-Velocity WCS, with
    ctype "OFFSET" for the spatial offset direction

    Parameters
    ----------
    wcs : :class:`~astropy.wcs.WCS`
        The WCS of the spectral cube. This should already be sanitized and
        have the spectral axis along the third dimension.
    spatial_scale: :class:`~astropy.units.Quantity`
        The spatial scale of the position axis

    Returns
    -------
    wcs_slice :class:`~astropy.wcs.WCS`
        The resulting WCS slice
    """

    # Extract spectral slice
    wcs_slice = wcs.sub([0, WCSSUB_SPECTRAL])

    # Set spatial parameters
    wcs_slice.wcs.crpix[0] = 1.
    wcs_slice.wcs.cdelt[0] = spatial_scale.to(u.degree).value
    wcs_slice.wcs.crval[0] = 0.
    wcs_slice.wcs.ctype[0] = "OFFSET"
    wcs_slice.wcs.cunit[0] = 'deg'
    # Not clear why this is needed, but apparently sub with 0 sets pc[1,0] = 1,
    # which is incorrect
    try:
        wcs_slice.wcs.pc[1,0] = wcs_slice.wcs.pc[0,1] = 0
    except AttributeError:
        pass

    return wcs_slice
