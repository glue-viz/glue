import logging

import numpy as np

__all__ = ['Coordinates', 'WCSCoordinates', 'WCSCube']


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
    distortions. Restricted to 2D images

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

    def pixel2world(self, xpix, ypix):
        '''
        Convert pixel to world coordinates, preserving input type/shape

        Parameters
        ----------
        xpix, ypix: scalars, lists, or Numpy arrays
            The pixel coordinates to convert

        Returns
        -------
        xworld, yworld: scalars, lists or Numpy arrays
            The corresponding world coordinates
        '''
        if type(xpix) is not type(ypix):
            if np.isscalar(xpix) and np.isscalar(ypix):
                pass
            else:
                raise TypeError("xpix and ypix types do not match: %s/%s" %
                                (type(xpix), type(ypix)))

        if np.isscalar(xpix):
            xworld, yworld = self._wcs.wcs_pix2sky(
                np.array([xpix]),
                np.array([ypix]), 1)
            return xworld[0], yworld[0]
        elif type(xpix) == list:
            xworld, yworld = self._wcs.wcs_pix2sky(np.array(xpix),
                                                   np.array(ypix), 1)
            return xworld.tolist(), yworld.tolist()
        elif isinstance(xpix, np.ndarray):
            logging.debug("xpix, ypix shapes: %s %s", xpix.shape, ypix.shape)

            xworld, yworld = self._wcs.wcs_pix2sky(xpix, ypix, 1)

            xworld.shape = xpix.shape
            yworld.shape = ypix.shape
            return xworld, yworld
        else:
            raise TypeError("Unexpected type for pixel coordinates: %s"
                            % type(xpix))

    def world2pixel(self, xworld, yworld):
        '''
        Convert pixel to world coordinates, preserving input type/shape

        Parameters
        ----------
        xworld, yworld: scalars, lists or Numpy arrays
            The world coordinates to convert

        Returns
        -------
        xpix, ypix: scalars, lists, or Numpy arrays
            The corresponding pixel coordinates
        '''

        if type(xworld) is not type(yworld):
            if np.isscalar(xworld) and np.isscalar(yworld):
                pass
            else:
                raise TypeError("xworld and yworld types do not match: %s/%s" %
                                (type(xworld), type(yworld)))

        if np.isscalar(xworld):
            xpix, ypix = self._wcs.wcs_sky2pix(np.array([xworld]),
                                               np.array([yworld]), 1)
            return xpix[0], ypix[0]
        elif type(xworld) == list:
            xpix, ypix = self._wcs.wcs_sky2pix(np.array(xworld),
                                               np.array(yworld), 1)
            return xpix.tolist(), ypix.tolist()
        elif isinstance(xworld, np.ndarray):
            xpix, ypix = self._wcs.wcs_sky2pix(xworld, yworld, 1)
            xpix.shape = xworld.shape
            ypix.shape = yworld.shape
            return xpix, ypix
        else:
            raise TypeError("Unexpected type for world coordinates: %s" %
                            type(xworld))

    def axis_label(self, axis):
        letters = ['y', 'x']
        header = self._header
        num = _get_ndim(header) - axis  # number orientation reversed
        key = 'CTYPE%i' % num
        if key in header:
            return 'World %s: %s' % (letters[axis], header[key])
        return 'World %s' % (letters[axis])


class WCSCubeCoordinates(WCSCoordinates):

    def __init__(self, header):
        super(WCSCubeCoordinates, self).__init__(header)
        from astropy.io import fits
        if not isinstance(header, fits.Header):
            raise TypeError("Header must by an astropy.io.fits.Header "
                            "instance")

        if _get_ndim(header) != 3:
            raise AttributeError("Header must describe a 3D array")

        self._header = header

        try:
            self._cdelt3 = header['CDELT3'] if 'CDELT3' in header \
                else header['CD3_3']
            self._crpix3 = header['CRPIX3']
            self._crval3 = header['CRVAL3']
        except KeyError:
            raise AttributeError("Input header must have CRPIX3, CRVAL3, and "
                                 "CDELT3 or CD3_3 keywords")
        self._ctype3 = ""
        if 'CTYPE3' in header:
            self._ctype3 = header['CTYPE3']

        # make sure there is no non-standard rotation
        keys = ['CD1_3', 'CD2_3', 'CD3_2', 'CD3_1']
        for k in keys:
            if k in header and header[k] != 0:
                raise AttributeError("Cannot handle non-zero keyword: "
                                     "%s = %s" %
                                     (k, header[k]))
        self._fix_header_for_2d()

        from astropy import wcs
        self._wcs = wcs.WCS(header)

    def _fix_header_for_2d(self):
        #workaround for astropy.wcs -- need to remove 3D header keywords
        self._header['NAXIS'] = 2
        for tag in ['NAXIS3', 'CDELT3', 'CD3_3', 'CRPIX3', 'CRVAL3', 'CTYPE3']:
            if tag in self._header:
                del self._header[tag]

    def pixel2world(self, xpix, ypix, zpix):
        xout, yout = WCSCoordinates.pixel2world(self, xpix, ypix)
        zout = (zpix - self._crpix3) * self._cdelt3 + self._crval3
        return xout, yout, zout

    def world2pixel(self, xworld, yworld, zworld):
        xout, yout = WCSCoordinates.world2pixel(self, xworld, yworld)
        zout = (zworld - self._crval3) / self._cdelt3 + self._crpix3
        return xout, yout, zout

    def axis_label(self, axis):
        letters = ['z', 'y', 'x']
        keys = ["", "", ""]
        if self._ctype3:
            keys[0] = ": %s" % self._ctype3
        if 'CTYPE2' in self._header:
            keys[1] = ": %s" % self._header['CTYPE2']
        if 'CTYPE1' in self._header:
            keys[2] = ": %s" % self._header['CTYPE1']
        return "World %s %s" % (letters[axis], keys[axis])


def coordinates_from_header(header):
    """ Convert a FITS header into a glue Coordinates object

    :param header: Header to convert
    :type header: :class:`astropy.io.fits.Header`

    :rtype: :class:`~glue.core.coordinates.Coordinates`
    """
    f = None
    ndim = _get_ndim(header)
    if ndim == 2:
        f = WCSCoordinates
    elif ndim == 3:
        f = WCSCubeCoordinates
    if f:
        try:
            return f(header)
        except AttributeError as e:
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
