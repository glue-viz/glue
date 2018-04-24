from __future__ import absolute_import, division, print_function

import logging

import numpy as np

from glue.utils import unbroadcast, broadcast_to, axis_correlation_matrix


__all__ = ['Coordinates', 'WCSCoordinates', 'coordinates_from_header', 'coordinates_from_wcs']


class Coordinates(object):

    '''
    Base class for coordinate transformation
    '''

    def __init__(self):
        pass

    def pixel2world(self, *args):
        """
        Convert pixel to world coordinates, preserving input type/shape.

        Parameters
        ----------
        *pixel : scalars lists, or Numpy arrays
            The pixel coordinates (0-based) to convert

        Returns
        -------
        *world : Numpy arrays
            The corresponding world coordinates
        """
        return args

    def world2pixel(self, *args):
        """
        Convert world to pixel coordinates, preserving input type/shape.

        Parameters
        ----------
        *world : scalars lists, or Numpy arrays
            The world coordinates to convert

        Returns
        -------
        *pixel : Numpy arrays
            The corresponding pixel coordinates
        """
        return args

    def default_world_coords(self, ndim):
        return np.zeros(ndim, dtype=float)

    # PY3: pixel2world_single_axis(self, *pixel, axis=None)

    def pixel2world_single_axis(self, *pixel, **kwargs):
        """
        Convert pixel to world coordinates, preserving input type/shape.

        This is a wrapper around pixel2world which returns the result for just
        one axis, and also determines whether the calculation can be sped up
        if broadcasting is present in the input arrays.

        Parameters
        ----------
        *pixel : scalars lists, or Numpy arrays
            The pixel coordinates (0-based) to convert
        axis : int, optional
            If only one axis is needed, it should be specified since the
            calculation will be much more efficient.

        Returns
        -------
        world : `numpy.ndarray`
            The world coordinates for the requested axis
        """

        # PY3: the following is needed for Python 2
        axis = kwargs.get('axis', None)

        if axis is None:
            raise ValueError("axis needs to be set")

        if np.size(pixel[0]) == 0:
            return np.array([], dtype=float)

        original_shape = pixel[0].shape
        pixel_new = []

        # NOTE: the axis passed to this function is the WCS axis not the Numpy
        # axis, so we need to convert it as needed.
        dep_axes = self.dependent_axes(len(pixel) - 1 - axis)
        for ip, p in enumerate(pixel):
            if (len(pixel) - 1 - ip) in dep_axes:
                pixel_new.append(unbroadcast(p))
            else:
                pixel_new.append(p.flat[0])
        pixel = np.broadcast_arrays(*pixel_new)

        result = self.pixel2world(*pixel)

        return broadcast_to(result[axis], original_shape)

    def world2pixel_single_axis(self, *world, **kwargs):
        """
        Convert world to pixel coordinates, preserving input type/shape.

        This is a wrapper around world2pixel which returns the result for just
        one axis, and also determines whether the calculation can be sped up
        if broadcasting is present in the input arrays.

        Parameters
        ----------
        *world : scalars lists, or Numpy arrays
            The world coordinates to convert
        axis : int, optional
            If only one axis is needed, it should be specified since the
            calculation will be much more efficient.

        Returns
        -------
        pixel : `numpy.ndarray`
            The pixel coordinates for the requested axis
        """

        # PY3: the following is needed for Python 2
        axis = kwargs.get('axis', None)

        if axis is None:
            raise ValueError("axis needs to be set")

        if np.size(world[0]) == 0:
            return np.array([], dtype=float)

        original_shape = world[0].shape
        world_new = []
        # NOTE: the axis passed to this function is the WCS axis not the Numpy
        # axis, so we need to convert it as needed.
        dep_axes = self.dependent_axes(len(world) - 1 - axis)
        for iw, w in enumerate(world):
            if (len(world) - 1 - iw) in dep_axes:
                world_new.append(unbroadcast(w))
            else:
                world_new.append(w.flat[0])
        world = np.broadcast_arrays(*world_new)

        result = self.world2pixel(*world)

        return broadcast_to(result[axis], original_shape)

    def world_axis(self, data, axis):
        """
        Find the world coordinates along a given dimension, and which for now
        we center on the pixel origin.

        Parameters
        ----------
        data : `~glue.core.data.Data`
            The data to compute the coordinate axis for (this is used to
            determine the size of the axis)
        axis : int
            The axis to compute, in Numpy axis order

        Notes
        -----
        This method computes the axis values using pixel positions at the
        center of the data along all other axes. This will therefore only give
        the correct result for non-dependent axes (which can be checked using
        the ``dependent_axes`` method).
        """
        pixel = []
        for i, s in enumerate(data.shape):
            if i == axis:
                pixel.append(np.arange(data.shape[axis]))
            else:
                pixel.append(np.repeat((s - 1) / 2, data.shape[axis]))
        return self.pixel2world_single_axis(*pixel[::-1],
                                            axis=data.ndim - 1 - axis)

    def world_axis_unit(self, axis):
        """
        Return the unit of the world coordinate given by ``axis`` (assuming the
        Numpy axis order)
        """
        return ''

    def axis_label(self, axis):
        return "World {}".format(axis)

    def dependent_axes(self, axis):
        """Return a tuple of which world-axes are non-independent
        from a given pixel axis

        The axis index is given in numpy ordering convention (note that
        opposite the fits convention)
        """
        return (axis,)

    def __gluestate__(self, context):
        return {}  # no state

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls()


class WCSCoordinates(Coordinates):

    '''
    Class for coordinate transformation based on the WCS FITS
    standard.  This class does not take into account
    distortions.

    Parameters
    ----------
    header : :class:`astropy.io.fits.Header`
        FITS header (derived from WCS if not given)
    wcs : :class:`astropy.wcs.WCS`
        WCS object to use, if different from header

    References
    ----------
    * Greisen & Calabretta (2002), Astronomy and Astrophysics, 395, 1061
    * Calabretta & Greisen (2002), Astronomy and Astrophysics, 395, 1077
    * Greisen, Calabretta, Valdes & Allen (2006), Astronomy and
      Astrophysics, 446, 747
    '''

    def __init__(self, header=None, wcs=None):
        super(WCSCoordinates, self).__init__()
        from astropy.wcs import WCS

        if header is None and wcs is None:
            raise ValueError('Must provide either FITS header or WCS or both')

        if header is None:
            header = wcs.to_header()

        self._header = header

        try:
            naxis = header['NAXIS']
        except (KeyError, TypeError):
            naxis = None

        wcs = wcs or WCS(header, naxis=naxis)

        # update WCS interface if using old API
        mapping = {'wcs_pix2world': 'wcs_pix2sky',
                   'wcs_world2pix': 'wcs_sky2pix',
                   'all_pix2world': 'all_pix2sky'}
        for k, v in mapping.items():
            if not hasattr(wcs, k):
                setattr(wcs, k, getattr(wcs, v))

        self._wcs = wcs

        # Pre-compute dependent axes. The matrix returned by
        # axis_correlation_matrix is (n_world, n_pixel) but we want to know
        # which pixel coordinates are linked to which other pixel coordinates.
        # So to do this we take a column from the matrix and find if there are
        # any entries in common with all other columns in the matrix.
        matrix = axis_correlation_matrix(wcs)[::-1, ::-1]
        self._dependent_axes = []
        for axis in range(wcs.naxis):
            world_dep = matrix[:, axis:axis + 1]
            dependent = tuple(np.nonzero((world_dep & matrix).any(axis=0))[0])
            self._dependent_axes.append(dependent)

    def world_axis_unit(self, axis):
        return str(self._wcs.wcs.cunit[self._wcs.naxis - 1 - axis])

    @property
    def wcs(self):
        return self._wcs

    @property
    def header(self):
        return self._header

    def dependent_axes(self, axis):
        return self._dependent_axes[axis]

    def __setstate__(self, state):
        self.__dict__ = state
        # wcs object doesn't seem to unpickle properly. reconstruct it
        from astropy.wcs import WCS
        try:
            naxis = self._header['NAXIS']
        except (KeyError, TypeError):
            naxis = None
        self._wcs = WCS(self._header, naxis=naxis)

    def pixel2world(self, *pixel):
        # PY3: can just do pix2world(*pixel, 0)
        if np.size(pixel[0]) == 0:
            return tuple(np.array([], dtype=float) for p in pixel)
        else:
            return self._wcs.wcs_pix2world(*(tuple(pixel) + (0,)))

    def world2pixel(self, *world):
        # PY3: can just do world2pix(*world, 0)
        if np.size(world[0]) == 0:
            return tuple(np.array([], dtype=float) for w in world)
        else:
            return self._wcs.wcs_world2pix(*(tuple(world) + (0,)))

    def default_world_coords(self, ndim):
        if ndim != self._wcs.naxis:
            raise ValueError("Requested default world coordinates for {0} "
                             "dimensions, WCS has {1}".format(ndim, self._wcs.naxis))
        return self._wcs.wcs.crval

    def axis_label(self, axis):
        header = self._header
        num = _get_ndim(header) - axis  # number orientation reversed
        ax = self._header.get('CTYPE%i' % num)
        if ax is not None:
            if len(ax) == 8 or '-' in ax:  # assume standard format
                ax = ax[:5].split('-')[0].title()
            else:
                ax = ax.title()

            translate = dict(
                Glon='Galactic Longitude',
                Glat='Galactic Latitude',
                Ra='Right Ascension',
                Dec='Declination',
                Velo='Velocity',
                Freq='Frequency'
            )
            return translate.get(ax, ax)
        return super(WCSCoordinates, self).axis_label(axis)

    def __gluestate__(self, context):
        return dict(header=self._wcs.to_header_string())

    @classmethod
    def __setgluestate__(cls, rec, context):
        from astropy.io import fits
        return cls(fits.Header.fromstring(rec['header']))


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
    return Coordinates()


def _get_ndim(header):
    if 'NAXIS' in header:
        return header['NAXIS']
    if 'WCSAXES' in header:
        return header['WCSAXES']
    return None


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
    from astropy.io import fits
    hdr_str = wcs.wcs.to_header()
    hdr = fits.Header.fromstring(hdr_str)
    try:
        return WCSCoordinates(hdr, wcs)
    except (AttributeError, TypeError) as e:
        print(e)
    return Coordinates()


def header_from_string(string):
    """
    Convert a string to a FITS header.
    """
    from astropy.io import fits
    return fits.Header.fromstring(string, sep='\n')
