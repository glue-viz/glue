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
        from ..external.astro import WCS

        self._header = header
        wcs = wcs or WCS(header)

        # update WCS interface if using old API
        mapping = {'wcs_pix2world': 'wcs_pix2sky',
                   'wcs_world2pix': 'wcs_sky2pix',
                   'all_pix2world': 'all_pix2sky'}
        for k, v in mapping.items():
            if not hasattr(wcs, k):
                setattr(wcs, k, getattr(wcs, v))

        self._wcs = wcs

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
        from ..external.astro import WCS
        self._wcs = WCS(self._header)

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
        from ..external.astro import fits
        return cls(fits.Header.fromstring(rec['header']))


def coordinates_from_header(header):
    """ Convert a FITS header into a glue Coordinates object

    :param header: Header to convert
    :type header: :class:`astropy.io.fits.Header`

    :rtype: :class:`~glue.core.coordinates.Coordinates`
    """
    try:
        return WCSCoordinates(header)
    except (AttributeError, TypeError, AssertionError) as e:
        print(e)
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
    from ..external.astro import fits
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
    from ..external.astro import fits
    cards = []
    for s in string.splitlines():
        try:
            l, r = s.split('=')
            key = l.strip()
            value = r.split('/')[0].strip()
            try:
                value = int(value)
            except ValueError:
                pass
        except ValueError:
            continue
        cards.append(fits.Card(key, value))
    return fits.Header(cards)
