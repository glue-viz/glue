from __future__ import absolute_import, division, print_function

# WCS utilities taken from spectral_cube project

__all__ = ['reindex_wcs', 'drop_axis']


def reindex_wcs(wcs, inds):
    """
    Re-index a WCS given indices.  The number of axes may be reduced.

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to be manipulated
    inds: np.array(dtype='int')
        The indices of the array to keep in the output.
        e.g. swapaxes: [0,2,1,3]
        dropaxes: [0,1,3]
    """
    from astropy.wcs import WCS
    wcs_parameters_to_preserve = ['cel_offset', 'dateavg', 'dateobs', 'equinox',
                                  'latpole', 'lonpole', 'mjdavg', 'mjdobs', 'name',
                                  'obsgeo', 'phi0', 'radesys', 'restfrq',
                                  'restwav', 'specsys', 'ssysobs', 'ssyssrc',
                                  'theta0', 'velangl', 'velosys', 'zsource']

    if not isinstance(inds, np.ndarray):
        raise TypeError("Indices must be an ndarray")

    if inds.dtype.kind != 'i':
        raise TypeError('Indices must be integers')

    outwcs = WCS(naxis=len(inds))
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    cdelt = wcs.wcs.get_cdelt()
    pc = wcs.wcs.get_pc()

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.cname = [wcs.wcs.cname[i] for i in inds]
    outwcs.wcs.pc = pc[inds[:, None], inds[None, :]]

    return outwcs


def drop_axis(wcs, dropax):
    """
    Drop the ax on axis dropax

    Remove an axis from the WCS
    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS with naxis to be chopped to naxis-1
    dropax: int
        The index of the WCS to drop, counting from 0 (i.e., python convention,
        not FITS convention)
    """
    inds = list(range(wcs.wcs.naxis))
    inds.pop(dropax)
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)
