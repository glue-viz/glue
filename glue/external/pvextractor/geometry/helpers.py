import numpy as np

from astropy import units as u
from astropy.coordinates import (SkyCoord, BaseCoordinateFrame,
                                 UnitSphericalRepresentation,
                                 CartesianRepresentation)

from .path import Path


class PathFromCenter(Path):
    """
    A simple path defined by a center, length, and position angle.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The center of the path
    length : `~astropy.units.Quantity`
        The length of the path in angular units
    angle : `~astropy.units.Quantity`
        The position angle of the path, counter-clockwise
    sample : int
        How many points to sample along the path. By default, this is 2 (the
        two end points. For small fields of view, this will be a good
        approximation to the path, but for larger fields of view, where
        spherical distortions become important, this should be increased to
        provide a smooth path.
    width : None or float or :class:`~astropy.units.Quantity`
        The width of the path. If ``coords`` is passed as a list of pixel
        positions, the width should be given (if passed) as a floating-point
        value in pixels. If ``coords`` is a coordinate object, the width
        should be passed as a :class:`~astropy.units.Quantity` instance with
        units of angle. If None, interpolation is used at the position of the
        path.

    Notes
    -----
    The orientation of the final path will be such that for a position angle of
    zero, the path goes from South to North. For a position angle of 90
    degrees, the path will go from West to East.
    """

    def __init__(self, center, length=None, angle=None, sample=2, width=None):

        # Check input types

        if not isinstance(center, (SkyCoord, BaseCoordinateFrame)):
            raise TypeError("The central position should be given as a SkyCoord object")

        if not isinstance(length, u.Quantity) or not length.unit.is_equivalent(u.deg):
            raise TypeError("The length should be given as an angular Quantity")

        if not isinstance(angle, u.Quantity) or not angle.unit.is_equivalent(u.deg):
            raise TypeError("The angle should be given as an angular Quantity")

        # We set up the path by adding and removing half the length along the
        # declination axis, then rotate the resulting two points around the
        # center.

        # Convert the central position to cartesian coordinates
        c1, c2, c3 = center.cartesian.xyz.value

        # Find the end points of the path
        clon, clat = center.spherical.lon, center.spherical.lat
        try:
            plat = clat + np.linspace(-length * 0.5, length * 0.5, sample)
        except ValueError:  # Numpy 1.10+
            plat = clat + np.linspace(-length.value * 0.5, length.value * 0.5, sample) * length.unit

        x, y, z = UnitSphericalRepresentation(clon, plat).to_cartesian().xyz.value

        # Rotate around central point

        # Because longitude increases to the left, we have to take -angle
        angle = -angle

        # We rotate (x,y,z) around (c1,c2,c3) by making use of the following
        # equations:
        #
        # x' = x cos a + (1 - cos a)(c1c1x + c1c2y + c1c3z) + (c2z - c3y)sin a
        # y' = y cos a + (1 - cos a)(c2c1x + c2c2y + c2c3z) + (c3x - c1z)sin a
        # z' = z cos a + (1 - cos a)(c3c1x + c3c2y + c3c3z) + (c1y - c2x)sin a
        #
        # Source: https://www.uwgb.edu/dutchs/MATHALGO/sphere0.htm

        cosa = np.cos(angle)
        sina = np.sin(angle)

        xd = x * cosa + (1 - cosa) * (c1*c1*x + c1*c2*y + c1*c3*z) + (c2 * z - c3 * y) * sina
        yd = y * cosa + (1 - cosa) * (c2*c1*x + c2*c2*y + c2*c3*z) + (c3 * x - c1 * z) * sina
        zd = z * cosa + (1 - cosa) * (c3*c1*x + c3*c2*y + c3*c3*z) + (c1 * y - c2 * x) * sina

        # Construct representations
        points = center.realize_frame(CartesianRepresentation(x=xd, y=yd, z=zd))

        super(PathFromCenter, self).__init__(points, width=width)
