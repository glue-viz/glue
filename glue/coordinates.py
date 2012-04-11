import pywcs
import numpy as np

import pyfits.core

class Coordinates(object):
    '''
    Base class for coordinate transformation
    '''

    def __init__(self):
        pass

    def pixel2world(self, x, y, z):
        return x, y

    def world2pixel(self, x, y, z):
        return x, y


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
        self._header = header
        self._wcs = pywcs.WCS(header)

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
            raise Exception("xpix and ypix types do not match: %s/%s" %
                            (type(xpix), type(ypix)))

        if np.isscalar(xpix):
            xworld, yworld = self._wcs.wcs_pix2sky(np.array([xpix]),
                                                  np.array([ypix]), 1)
            return xworld[0], yworld[0]
        elif type(xpix) == list:
            xworld, yworld = self._wcs.wcs_pix2sky(np.array(xpix),
                                                  np.array(ypix), 1)
            return xworld.tolist(), yworld.tolist()
        elif isinstance(xpix, np.ndarray):
            return self._wcs.wcs_pix2sky(xpix, ypix, 1)
        else:
            raise Exception("Unexpected type for pixel coordinates: %s"
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
            raise Exception("xworld and yworld types do not match: %s/%s" %
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
            return self._wcs.wcs_sky2pix(xworld, yworld, 1)
        else:
            raise Exception("Unexpected type for world coordinates: %s" %
                            type(xworld))

class WCSCubeCoordinates(WCSCoordinates):

    def __init__(self, header):
        if not isinstance(header, pyfits.core.Header):
            raise TypeError("Header must by a pyfits header instance")

        if 'NAXIS' not in header or header['NAXIS'] != 3:
            raise AttributeError("Header must describe a 3D array")

        self._header = header
        self._wcs = pywcs.WCS(header)

        try:
            self._cdelt3 = header['CDELT3'] if 'CDELT3' in header else header['CD3_3']
            self._crpix3 = header['CRPIX3']
            self._crval3 = header['CRVAL3']
        except KeyError:
            raise AttributeError("Input header must have CRPIX3, CRVAL3, and "
                             "CDELT3 or CD3_3 keywords")

        # make sure there is no non-standard rotation
        keys = ['CD1_3', 'CD2_3', 'CD3_2', 'CD3_1']
        for k in keys:
            if k in header and header[k] != 0:
                raise AttributeError("Cannot handle non-zero keyword: %s = %s" %
                                     (k, header[k]))

    def pixel2world(self, xpix, ypix, zpix):
        xout, yout = WCSCoordinates.pixel2world(self, xpix, ypix)
        zout = (zpix - self._crpix3) * self._cdelt3 + self._crval3
        return xout, yout, zout

    def world2pixel(self, xworld, yworld, zworld):
        xout, yout = WCSCoordinates.world2pixel(self, xworld, yworld)
        zout = (zworld - self._crval3) / self._cdelt3 + self._crpix3
        return xout, yout, zout


