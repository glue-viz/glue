import abc
import logging

import numpy as np

from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseLowLevelWCS

__all__ = ['Coordinates', 'IdentityCoordinates', 'AffineCoordinates',
           'coordinates_from_header', 'coordinates_from_wcs', 'header_from_string']


def default_world_coords(wcs):
    if isinstance(wcs, WCS):
        return wcs.wcs.crval
    else:
        return np.zeros(wcs.world_n_dim, dtype=float)


# The Coordinates class (and sub-classes) are used to map from pixel to world
# coordinates (and vice-versa). Rather than re-invent a new API, we adopt the
# one from the Astropy project which is generic and not specific to astronomy,
# which then allows us to make use of visualization tools (e.g. WCSAxes) built
# around this API.


class Coordinates(BaseLowLevelWCS, metaclass=abc.ABCMeta):
    """
    Base class for coordinate transformation
    """

    def __gluestate__(self, context):
        return {}  # no state

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls()

    # Kept for backward-compatibility

    def pixel2world(self, *args):
        return self.pixel_to_world_values(*args)

    def world2pixel(self, *args):
        return self.world_to_pixel_values(*args)

    def world_axis_unit(self, axis):
        return self.world_axis_units[self.world_n_dim - axis - 1]


class WCSCoordinates(WCS):

    def pixel2world(self, *args):
        return self.pixel_to_world_values(*args)

    def world2pixel(self, *args):
        return self.world_to_pixel_values(*args)

    def world_axis_unit(self, axis):
        return self.world_axis_units[self.world_n_dim - axis - 1]


class IdentityCoordinates(WCSCoordinates):

    def __init__(self, ndim):
        super().__init__(naxis=ndim)

    def __gluestate__(self, context):
        return {'ndim': self.pixel_n_dim}

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(ndim=rec['ndim'])


class AffineCoordinates(WCSCoordinates):
    """
    Coordinates determined via an affine transformation represented by an
    augmented matrix of shape N+1 x N+1 matrix, where N is the number of pixel
    and world coordinates. The last column of the matrix should be used for
    the translation term, and the last row should be set to 0 except for the
    last column which should be 1.

    Note that the order of the dimensions in the matrix (x, y) should be the
    opposite of the order of the order of dimensions of Numpy arrays (y, x).
    """

    # Note that for now the easiest way to implement this is to sub-class
    # WCS, which means that this will automatically work with WCSAxes. In
    # future it would be good to make this independent from WCS but it will
    # require changes to WCSAxes.

    def __init__(self, matrix, units=None, labels=None):

        if matrix.ndim != 2:
            raise ValueError("Affine matrix should be two-dimensional")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Affine matrix should be square")

        if np.any(matrix[-1, :-1] != 0) or matrix[-1, -1] != 1:
            raise ValueError("Last row of matrix should be zeros and a one")

        if units is not None and len(units) != matrix.shape[0] - 1:
            raise ValueError("Expected {0} units, got {1}".format(matrix.shape[0] - 1, len(units)))

        if labels is not None and len(labels) != matrix.shape[0] - 1:
            raise ValueError("Expected {0} labels, got {1}".format(matrix.shape[0] - 1, len(labels)))

        self.matrix = matrix
        self.units = units
        self.labels = labels

        super().__init__(naxis=self.matrix.shape[0] - 1)
        self.wcs.cd = self.matrix[:-1, :-1]
        self.wcs.crpix = np.ones(self.naxis)
        self.wcs.crval = self.matrix[:-1, -1]
        if labels is not None:
            self.wcs.ctype = labels
        if units is not None:
            self.wcs.cunit = units

    def __gluestate__(self, context):
        return dict(matrix=context.do(self.matrix),
                    labels=self.labels,
                    units=self.units)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['matrix']),
                   units=rec['units'],
                   labels=rec['labels'])


def coordinates_from_header(header):
    """
    Convert a FITS header into a glue Coordinates object.

    Parameters
    ----------
    header : :class:`astropy.io.fits.Header`
        Header to convert

    Returns
    -------
    coordinates : :class:`~glue.core.coordinates.Coordinates`
    """

    # We check whether the header contains at least CRVAL1 - if not, we would
    # end up with a default WCS that isn't quite 1 to 1 (because of a 1-pixel
    # offset) so better use Coordinates in that case.

    from astropy.io.fits import Header

    if isinstance(header, Header) and 'CRVAL1' in header:
        try:
            return WCSCoordinates(header)
        except Exception as e:
            logging.getLogger(__name__).warn(
                "\n\n*******************************\n"
                "Encounted an error during WCS parsing. "
                "Discarding world coordinates! "
                "\n{}\n"
                "*******************************\n\n".format(str(e)))
    try:
        return IdentityCoordinates(header.get('NAXIS', 2))
    except Exception:
        return IdentityCoordinates(2)


def coordinates_from_wcs(wcs):
    """
    Convert an Astropy WCS object into a glue Coordinates object.

    Parameters
    ----------
    wcs : :class:`astropy.wcs.WCS`
        The WCS object to use

    Returns
    -------
    coordinates : :class:`~glue.core.coordinates.Coordinates`
    """
    return WCSCoordinates(wcs.to_header())


def header_from_string(string):
    """
    Convert a string to a FITS header.
    """
    from astropy.io import fits
    return fits.Header.fromstring(string, sep='\n')
