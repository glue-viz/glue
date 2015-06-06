from __future__ import print_function

import sys

import numpy as np
from astropy.wcs import WCSSUB_CELESTIAL

try:
    from astropy.coordinates import BaseCoordinateFrame
except ImportError:  # astropy <= 0.3
    from astropy.coordinates import SphericalCoordinatesBase as BaseCoordinateFrame

from ..utils.wcs_utils import get_wcs_system_frame, get_spatial_scale

class Polygon(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def segment_angles(x, y):

    dx = np.diff(x)
    dy = np.diff(y)

    d = np.hypot(dx, dy)

    cos_theta = (-dx[:-1] * dx[1:] - dy[:-1] * dy[1:]) / (d[:-1] * d[1:])
    cos_theta = np.clip(cos_theta, -1., 1.)

    sin_theta = (-dx[:-1] * dy[1:] + dy[:-1] * dx[1:]) / (d[:-1] * d[1:])
    sin_theta = np.clip(sin_theta, -1., 1.)

    theta = np.arctan2(sin_theta, cos_theta)

    theta[0] = np.pi
    theta[-1] = np.pi

    return theta


def get_endpoints(x, y, width):

    # Pad with same values at ends, to find slope of perpendicular end
    # lines.
    try:
        xp = np.pad(x, 1, mode='edge')
        yp = np.pad(y, 1, mode='edge')
    except AttributeError:  # Numpy < 1.7
        xp = np.hstack([x[0], x, x[-1]])
        yp = np.hstack([y[0], y, y[-1]])

    dx = np.diff(xp)
    dy = np.diff(yp)

    alpha = segment_angles(xp, yp) / 2.
    beta = np.arctan2(dy, dx)[:-1]
    beta[0] = beta[1]
    gamma = -(np.pi - alpha - beta)

    dx = np.cos(gamma)
    dy = np.sin(gamma)

    angles = segment_angles(xp, yp) / 2.

    # Find points offset from main curve, on bisecting lines
    x1 = x - dx * width * 0.5 / np.sin(angles)
    x2 = x + dx * width * 0.5 / np.sin(angles)
    y1 = y - dy * width * 0.5 / np.sin(angles)
    y2 = y + dy * width * 0.5 / np.sin(angles)

    return x1, y1, x2, y2


class Path(object):
    """
    A curved path that may have a non-zero width and is used to extract
    slices from cubes.

    Parameters
    ----------
    xy_or_coords : list or Astropy coordinates
        The points defining the path. This can be passed as a list of (x, y)
        tuples, which is interpreted as being pixel positions, or it can be
        an Astropy coordinate object containing an array of 2 or more
        coordinates.
    width : None or float or :class:`~astropy.units.Quantity`
        The width of the path. If ``coords`` is passed as a list of pixel
        positions, the width should be given (if passed) as a floating-point
        value in pixels. If ``coords`` is a coordinate object, the width
        should be passed as a :class:`~astropy.units.Quantity` instance with
        units of angle.
    """

    def __init__(self, xy_or_coords, width=None):
        if isinstance(xy_or_coords, list):
            self._xy = xy_or_coords
            self._coords = None
        elif sys.version_info[0] > 2 and isinstance(xy_or_coords, zip):
            self._xy = list(xy_or_coords)
            self._coords = None
        else:
            self._xy = None
            self._coords = xy_or_coords
        self.width = width

    def add_point(self, xy_or_coord):
        """
        Add a point to the path

        Parameters
        ----------
        xy_or_coord : tuple or Astropy coordinate
            A tuple (x, y) containing the coordinates of the point to add (if
            the path is defined in pixel space), or an Astropy coordinate
            object (if it is defined in world coordinates).
        """
        if self._xy is not None:
            if isinstance(xy_or_coord, tuple):
                self._xy.append(xy_or_coord)
            else:
                raise TypeError("Path is defined as a list of pixel "
                                "coordinates, so `xy_or_coord` should be "
                                "a tuple of `(x,y)` pixel coordinates.")
        else:
            if isinstance(xy_or_coord, BaseCoordinateFrame):
                raise NotImplementedError("Cannot yet append world coordinates to path")
            else:
                raise TypeError("Path is defined in world coordinates, "
                                "so `xy_or_coord` should be an Astropy "
                                "coordinate object.")

    def get_xy(self, wcs=None):
        """
        Return the pixel coordinates of the path.

        If the path is defined in world coordinates, the appropriate WCS
        transformation should be passed.

        Parameters
        ----------
        wcs : :class:`~astropy.wcs.WCS`
            The WCS transformation to assume in order to transform the path
            to pixel coordinates.
        """
        if self._xy is not None:
            return self._xy
        else:
            if wcs is None:
                raise ValueError("`wcs` is needed in order to compute "
                                 "the pixel coordinates")
            else:

                # Extract the celestial component of the WCS
                wcs_sky = wcs.sub([WCSSUB_CELESTIAL])

                # Find the astropy name for the coordinates
                # TODO: return a frame class with Astropy 0.4, since that can
                # also contain equinox/epoch info.
                celestial_system = get_wcs_system_frame(wcs_sky)

                world_coords = self._coords.transform_to(celestial_system)

                try:
                    xw, yw = world_coords.spherical.lon.degree, world_coords.spherical.lat.degree
                except AttributeError:  # astropy <= 0.3
                    xw, yw = world_coords.lonangle.degree, world_coords.latangle.degree

                return list(zip(*wcs_sky.wcs_world2pix(xw, yw, 0)))

    def sample_points_edges(self, spacing, wcs=None):

        x, y = zip(*self.get_xy(wcs=wcs))

        # Find the distance interval between all pairs of points
        dx = np.diff(x)
        dy = np.diff(y)
        dd = np.hypot(dx, dy)

        # Find the total displacement along the broken curve
        d = np.hstack([0., np.cumsum(dd)])

        # Figure out the number of points to sample, and stop short of the
        # last point.
        n_points = np.floor(d[-1] / spacing)

        if n_points == 0:
            raise ValueError("Path is shorter than spacing")

        d_sampled = np.linspace(0., n_points * spacing, n_points + 1)

        x_sampled = np.interp(d_sampled, d, x)
        y_sampled = np.interp(d_sampled, d, y)

        return d_sampled, x_sampled, y_sampled

    def sample_points(self, spacing, wcs=None):

        d_sampled, x_sampled, y_sampled = self.sample_points_edges(spacing, wcs=wcs)

        x_sampled = 0.5 * (x_sampled[:-1] + x_sampled[1:])
        y_sampled = 0.5 * (y_sampled[:-1] + y_sampled[1:])

        return x_sampled, y_sampled

    def sample_polygons(self, spacing, wcs=None):

        x, y = zip(*self.get_xy(wcs=wcs))

        d_sampled, x_sampled, y_sampled = self.sample_points_edges(spacing, wcs=wcs)

        # Find the distance interval between all pairs of points
        dx = np.diff(x)
        dy = np.diff(y)
        dd = np.hypot(dx, dy)

        # Normalize to find unit vectors
        dx = dx / dd
        dy = dy / dd

        # Find the total displacement along the broken curve
        d = np.hstack([0., np.cumsum(dd)])

        interval = np.searchsorted(d, d_sampled) - 1
        interval[0] = 0

        dx = dx[interval]
        dy = dy[interval]

        polygons = []

        x_beg = x_sampled - dx * spacing * 0.5
        x_end = x_sampled + dx * spacing * 0.5

        y_beg = y_sampled - dy * spacing * 0.5
        y_end = y_sampled + dy * spacing * 0.5

        if hasattr(self.width, 'unit'):
            scale = get_spatial_scale(wcs)
            width = (self.width / scale).decompose()
        else:
            width = self.width

        x1 = x_beg - dy * width * 0.5
        y1 = y_beg + dx * width * 0.5

        x2 = x_end - dy * width * 0.5
        y2 = y_end + dx * width * 0.5

        x3 = x_end + dy * width * 0.5
        y3 = y_end - dx * width * 0.5

        x4 = x_beg + dy * width * 0.5
        y4 = y_beg - dx * width * 0.5

        for i in range(len(x_sampled) - 1):
            p = Polygon([x1[i], x2[i], x3[i], x4[i]], [y1[i], y2[i], y3[i], y4[i]])
            polygons.append(p)

        return polygons

    def to_patches(self, spacing, **kwargs):
        from matplotlib.patches import Polygon as MPLPolygon
        patches = []
        for poly in self.sample_polygons(spacing):
            patches.append(MPLPolygon(zip(poly.x, poly.y), **kwargs))
        return patches
