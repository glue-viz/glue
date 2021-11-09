from astropy.wcs import WCS


def get_identity_wcs(naxis):
    """
    Create a WCS object with an identity transform.

    Parameters
    ----------
    naxis : float
        The number of axes.

    Returns
    -------
    `astropy.wcs.WCS`
        A WCS object with an identity transform.
    """
    wcs = WCS(naxis=naxis)
    wcs.wcs.ctype = ['X'] * naxis
    wcs.wcs.crval = [0.] * naxis
    wcs.wcs.crpix = [1.] * naxis
    wcs.wcs.cdelt = [1.] * naxis
    return wcs
