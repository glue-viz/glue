import logging

import numpy as np

__all__ = ['Coordinates', 'WCSCoordinates']


class Coordinates(object):
    '''
    Base class for coordinate transformation
    '''

    def __init__(self):
        pass

    def pixel2world(self, *args):
        return args

    def world2pixel(self, *args):
        return args

    def axis_label(self, axis):
        return "World %i" % axis


class WCSCoordinates(Coordinates):
    '''
    Class for coordinate transformation based on the WCS FITS
    standard.  This class does not take into account
    distortions.

    References
    ----------
      * Greisen & Calabretta (2002), Astronomy and Astrophysics, 395, 1061
      * Calabretta & Greisen (2002), Astronomy and Astrophysics, 395, 1077
      * Greisen, Calabretta, Valdes & Allen (2006), Astronomy and
        Astrophysics, 446, 747
    '''

    def __init__(self, header):
        super(WCSCoordinates, self).__init__()
        from astropy import wcs
        self._header = header
        self._wcs = wcs.WCS(header)

    @property
    def wcs(self):
        return self._wcs

    def __setstate__(self, state):
        self.__dict__ = state
        # wcs object doesn't seem to unpickle properly. reconstruct it
        from astropy import wcs
        self._wcs = wcs.WCS(self._header)

    def pixel2world(self, *pixel):
        '''
        Convert pixel to world coordinates, preserving input type/shape

        :param args: xpix, ypix[, zpix]: scalars, lists, or Numpy arrays
                     The pixel coordinates to convert

        *Returns*

        xworld, yworld, [zworld]: scalars, lists or Numpy arrays
            The corresponding world coordinates
        '''
        arrs = [np.asarray(p) for p in pixel]
        pix = np.vstack(a.ravel() for a in arrs).T
        result = tuple(self._wcs.wcs_pix2world(pix, 1).T)
        for r, a in zip(result, arrs):
            r.shape = a.shape
        return result

    def world2pixel(self, *world):
        '''
        Convert pixel to world coordinates, preserving input type/shape

        :param world:
            xworld, yworld[, zworld] : scalars, lists or Numpy arrays
            The world coordinates to convert

        *Returns*

        xpix, ypix: scalars, lists, or Numpy arrays
            The corresponding pixel coordinates
        '''
        arrs = [np.asarray(w) for w in world]
        pix = np.vstack(a.ravel() for a in arrs).T
        result = tuple(self._wcs.wcs_world2pix(pix, 1).T)
        for r, a in zip(result, arrs):
            r.shape = a.shape
        return result

    def _2d_axis_label(self, axis):
        letters = ['y', 'x']
        header = self._header
        num = _get_ndim(header) - axis  # number orientation reversed
        key = 'CTYPE%i' % num
        if key in header:
            return 'World %s: %s' % (letters[axis], header[key])
        return 'World %s' % (letters[axis])

    def _3d_axis_label(self, axis):
        letters = ['z', 'y', 'x']
        keys = ["", "", ""]
        if 'CTYPE3' in self._header:
            keys[0] = ": %s" % self._header['CTYPE3']
        if 'CTYPE2' in self._header:
            keys[1] = ": %s" % self._header['CTYPE2']
        if 'CTYPE1' in self._header:
            keys[2] = ": %s" % self._header['CTYPE1']
        return "World %s%s" % (letters[axis], keys[axis])

    def axis_label(self, axis):
        header = self._header
        if _get_ndim(header) == 2:
            return self._2d_axis_label(axis)
        elif _get_ndim(header) == 3:
            return self._3d_axis_label(axis)
        else:
            return super(WCSCoordinates, self).axis_label(axis)


def coordinates_from_header(header):
    """ Convert a FITS header into a glue Coordinates object

    :param header: Header to convert
    :type header: :class:`astropy.io.fits.Header`

    :rtype: :class:`~glue.core.coordinates.Coordinates`
    """
    try:
        return WCSCoordinates(header)
    except (AttributeError, TypeError) as e:
        print e
        pass
    return Coordinates()


def _get_ndim(header):
    if 'NAXIS' in header:
        return header['NAXIS']
    if 'WCSAXES' in header:
        return header['WCSAXES']
    return None


def coordinates_from_wcs(wcs):
    """Convert a wcs object into a glue Coordinates object

    :param wcs: The WCS object to use
    :rtype: :class:`~glue.core.coordinates.Coordinates`
    """
    from astropy.io import fits
    hdr_str = wcs.wcs.to_header()
    hdr = fits.Header.fromstring(hdr_str)
    return coordinates_from_header(hdr)
