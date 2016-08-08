from __future__ import absolute_import, division, print_function

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

    def world_axis(self, data, axis):
        """
        Find the world coordinates along a given dimension, and which for now we
        center on the pixel origin.

        Parameters
        ----------
        data : `~glue.core.data.Data`
            The data to compute the coordinate axis for (this is used to
            determine the size of the axis)
        axis : int
            The axis to compute, in Numpy axis order

        Notes
        -----
        This method computes the axis values using pixel positions at the center
        of the data along all other axes. This will therefore only give the
        correct result for non-dependent axes (which can be checked using the
        ``dependent_axes`` method)
        """
        pixel = []
        for i, s in enumerate(data.shape):
            if i == axis:
                pixel.append(np.arange(data.shape[axis]))
            else:
                pixel.append(np.repeat((s - 1) / 2, data.shape[axis]))
        return self.pixel2world(*pixel[::-1])[::-1][axis]

    def world_axis_unit(self, axis):
        """
        Return the unit of the world coordinate given by ``axis`` (assuming the
        Numpy axis order)
        """
        return ''

    def axis_label(self, axis):
        return "World %i" % axis

    def dependent_axes(self, axis):
        """Return a tuple of which world-axes are non-indepndent
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

    References
    ----------
      * Greisen & Calabretta (2002), Astronomy and Astrophysics, 395, 1061
      * Calabretta & Greisen (2002), Astronomy and Astrophysics, 395, 1077
      * Greisen, Calabretta, Valdes & Allen (2006), Astronomy and
        Astrophysics, 446, 747
    '''

    def __init__(self, header, wcs=None):
        super(WCSCoordinates, self).__init__()
        from astropy.wcs import WCS

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

    def world_axis_unit(self, axis):
        return str(self._wcs.wcs.cunit[self._wcs.naxis - 1 - axis])

    @property
    def wcs(self):
        return self._wcs

    @property
    def header(self):
        return self._header

    def dependent_axes(self, axis):
        # if distorted, all bets are off
        try:
            if any([self._wcs.sip, self._wcs.det2im1, self._wcs.det2im2]):
                return tuple(range(self._wcs.naxis))
        except AttributeError:
            pass

        # here, axis is the index number in numpy convention
        # we flip with [::-1] because WCS and numpy index
        # conventions are reversed
        pc = np.array(self._wcs.wcs.get_pc()[::-1, ::-1])
        ndim = pc.shape[0]
        pc[np.eye(ndim, dtype=np.bool)] = 0
        axes = self._wcs.get_axis_types()[::-1]

        # axes rotated
        if pc[axis, :].any() or pc[:, axis].any():
            return tuple(range(ndim))

        # XXX can spectral still couple with other axes by this point??
        if axes[axis].get('coordinate_type') != 'celestial':
            return (axis,)

        # in some cases, even the celestial coordinates are
        # independent. We don't catch that here.
        return tuple(i for i, a in enumerate(axes) if
                     a.get('coordinate_type') == 'celestial')

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
        result = tuple(self._wcs.wcs_pix2world(pix, 0).T)
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
        result = tuple(self._wcs.wcs_world2pix(pix, 0).T)
        for r, a in zip(result, arrs):
            r.shape = a.shape
        return result

    def axis_label(self, axis):
        header = self._header
        ndim = _get_ndim(header)
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
    """ Convert a FITS header into a glue Coordinates object

    :param header: Header to convert
    :type header: :class:`astropy.io.fits.Header`

    :rtype: :class:`~glue.core.coordinates.Coordinates`
    """

    # We check whether the header contains at least CRVAL1 - if not, we would
    # end up with a default WCS that isn't quite 1 to 1 (because of a 1-pixel
    # offset) so better use Coordinates in that case.

    from astropy.io.fits import Header

    if isinstance(header, Header) and 'CRVAL1' in header:
        try:
            return WCSCoordinates(header)
        except Exception as e:
            logging.getLogger(__name__).warn("\n\n*******************************\n"
                                             "Encounted an error during WCS parsing. "
                                             "Discarding world coordinates! "
                                             "\n%s\n"
                                             "*******************************\n\n" % e
                                             )
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
    try:
        return WCSCoordinates(hdr, wcs)
    except (AttributeError, TypeError) as e:
        print(e)
    return Coordinates()


def header_from_string(string):
    """
    Convert a string to a FITS header
    """
    from astropy.io import fits
    return fits.Header.fromstring(string, sep='\n')
