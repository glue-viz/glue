import pywcs
import numpy as np


class Coordinates(object):
    '''
    Base class for coordinate transformation
    '''

    def __init__(self):
        pass

    def pixel2world(self, x, y):
        return x, y

    def world2pixel(self, x, y):
        return x, y


class WCSCoordinates(Coordinates):
    '''
    Class for coordinate transformation based on the WCS FITS standard.
    This class does not take into account distortions.

    References
    ----------

      * Greisen & Calabretta (2002), Astronomy and Astrophysics, 395, 1061
      * Calabretta & Greisen (2002), Astronomy and Astrophysics, 395, 1077
      * Greisen, Calabretta, Valdes & Allen (2006), Astronomy and
        Astrophysics, 446, 747
    '''

    def __init__(self, header):
        self.wcs = pywcs.WCS(header)

    def pixel2world(self, xpix, ypix):
        '''
        Convert pixel to world coordinates, preserving input type/shape
        '''

        if type(xpix) is not type(ypix):
            raise Exception("xpix and ypix types do not match: %s/%s" % (type(xpix), type(ypix)))

        if np.isscalar(xpix):
            xworld, yworld = self.wcs.wcs_pix2sky(np.array([xpix]),
                                                  np.array([ypix]), 1)
            return xworld[0], yworld[0]
        elif type(xpix) == list:
            xworld, yworld = self.wcs.wcs_pix2sky(np.array(xpix),
                                                  np.array(ypix), 1)
            return xworld.tolist(), yworld.tolist()
        elif np.isarray(xpix):
            return self.wcs.wcs_pix2sky(xpix, ypix, 1)
        else:
            raise Exception("Unexpected type for pixel coordinates: %s" % type(xpix))

    def world2pixel(self, xworld, yworld):
        '''
        Convert world to pixel coordinates, preserving input type/shape
        '''

        if type(xworld) is not type(yworld):
            raise Exception("xworld and yworld types do not match: %s/%s" % (type(xworld), type(yworld)))

        if np.isscalar(xworld):
            xpix, ypix = self.wcs.wcs_sky2pix(np.array([xworld]),
                                              np.array([yworld]), 1)
            return xpix[0], ypix[0]
        elif type(xworld) == list:
            xpix, ypix = self.wcs.wcs_sky2pix(np.array(xworld),
                                              np.array(yworld), 1)
            return xpix.tolist(), ypix.tolist()
        elif np.isarray(xworld):
            return self.wcs.wcs_sky2pix(xworld, yworld, 1)
        else:
            raise Exception("Unexpected type for world coordinates: %s" % type(xworld))
