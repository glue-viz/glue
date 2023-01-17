import abc
import uuid
import logging

import numpy as np

from astropy.units import Quantity
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
# around this API. The API is defined/described in
# https://github.com/astropy/astropy-APEs/blob/master/APE14.rst


class Coordinates(BaseLowLevelWCS, metaclass=abc.ABCMeta):
    """
    Base class for coordinate transformation
    """

    def __init__(self, n_dim=None, pixel_n_dim=None, world_n_dim=None):
        self._pixel_n_dim = n_dim or pixel_n_dim
        self._world_n_dim = n_dim or world_n_dim
        self._world_uuids = [str(uuid.uuid4()) for i in range(self._world_n_dim)]

    @property
    def pixel_n_dim(self):
        return self._pixel_n_dim

    @property
    def world_n_dim(self):
        return self._world_n_dim

    @property
    def world_axis_physical_types(self):
        return [None] * self._world_n_dim

    @property
    def world_axis_units(self):
        return [''] * self._world_n_dim

    def array_index_to_world_values(self, *index_arrays):
        return self.pixel_to_world_values(*index_arrays[::-1])

    def world_to_array_index_values(self, *world_arrays):
        # Compatibility code, remove once we require Astropy 4.1 or later
        pixel_arrays = self.world_to_pixel_values(*world_arrays)
        if self.pixel_n_dim == 1:
            pixel_arrays = (pixel_arrays,)
        else:
            pixel_arrays = pixel_arrays[::-1]
        array_indices = tuple(np.asarray(np.floor(pixel + 0.5), dtype=np.int_) for pixel in pixel_arrays)
        return array_indices[0] if self.pixel_n_dim == 1 else array_indices

    @property
    def world_axis_object_components(self):
        return [(uid, 0, 'value') for uid in self._world_uuids]

    @property
    def world_axis_object_classes(self):
        return {self._world_uuids[idx]: (Quantity, (),
                                         {'unit': self.world_axis_units[idx]})
                for idx in range(self._world_n_dim)}

    def __gluestate__(self, context):
        return {}  # no state

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls()


class LegacyCoordinates(Coordinates):

    def __init__(self):
        super().__init__(pixel_n_dim=10, world_n_dim=10)

    def pixel_to_world_values(self, *pixel):
        if len(pixel) == 1:
            return pixel[0]
        else:
            return pixel

    def world_to_pixel_values(self, *world):
        if len(world) == 1:
            return world[0]
        else:
            return world


class IdentityCoordinates(Coordinates):

    def __init__(self, n_dim=None):
        super().__init__(pixel_n_dim=n_dim, world_n_dim=n_dim)

    def pixel_to_world_values(self, *pixel):
        if self.pixel_n_dim == 1:
            return pixel[0]
        else:
            return pixel

    def world_to_pixel_values(self, *world):
        if self.world_n_dim == 1:
            return world[0]
        else:
            return world

    @property
    def axis_correlation_matrix(self):
        return np.identity(self.pixel_n_dim).astype(bool)

    def __gluestate__(self, context):
        return {'ndim': self.pixel_n_dim}

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(n_dim=rec['ndim'])


class AffineCoordinates(Coordinates):
    """
    Coordinates determined via an affine transformation represented by an
    augmented matrix of shape N+1 x N+1 matrix, where N is the number of pixel
    and world coordinates. The last column of the matrix should be used for
    the translation term, and the last row should be set to 0 except for the
    last column which should be 1.

    Note that the order of the dimensions in the matrix (x, y) should be the
    opposite of the order of the order of dimensions of Numpy arrays (y, x).
    """

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

        self._matrix = matrix
        self._matrix_inv = np.linalg.inv(matrix)
        self._units = units
        self._labels = labels

        super().__init__(pixel_n_dim=self._matrix.shape[0] - 1,
                         world_n_dim=self._matrix.shape[0] - 1)

    def pixel_to_world_values(self, *pixel):
        scalar = np.all([np.isscalar(p) for p in pixel])
        pixel = np.array(np.broadcast_arrays(*(list(pixel) + [np.ones(np.shape(pixel[0]))])))
        pixel = np.moveaxis(pixel, 0, -1)
        world = np.matmul(pixel, self._matrix.T)
        world = tuple(np.moveaxis(world, -1, 0))[:-1]
        if self._matrix.shape[0] == 2:  # 1D
            return world[0]
        else:
            return world

    def world_to_pixel_values(self, *world):
        scalar = np.all([np.isscalar(w) for w in world])
        world = np.array(np.broadcast_arrays(*(list(world) + [np.ones(np.shape(world[0]))])))
        world = np.moveaxis(world, 0, -1)
        pixel = np.matmul(world, self._matrix_inv.T)
        pixel = tuple(np.moveaxis(pixel, -1, 0))[:-1]
        if self._matrix.shape[0] == 2:  # 1D
            return pixel[0]
        else:
            return pixel

    @property
    def world_axis_names(self):
        return self._labels or [''] * self.world_n_dim

    @property
    def world_axis_units(self):
        return self._units or [''] * self.world_n_dim

    def __gluestate__(self, context):
        return dict(matrix=context.do(self._matrix),
                    labels=self._labels,
                    units=self._units)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['matrix']),
                   units=rec['units'],
                   labels=rec['labels'])


# Kept for backward-compatibility
WCSCoordinates = WCS


def coordinates_from_header(header, hdulist=None):
    """
    Convert a FITS header into a glue Coordinates object.

    Parameters
    ----------
    header : :class:`astropy.io.fits.Header`
        Header to convert

    hdulist : :class:`astropy.io.fits.HDUList`, optional
        The full HDUList is required when header keywords point to a
        'Distortion Paper' lookup table stored in a different extension.

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
            return WCSCoordinates(header, hdulist)
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
