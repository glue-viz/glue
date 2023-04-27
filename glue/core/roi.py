import copy
import warnings

import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle, Path as MplPath, PathPatch
from matplotlib.transforms import IdentityTransform, blended_transform_factory

from glue.core.component import CategoricalComponent
from glue.core.exceptions import UndefinedROI
from glue.utils import points_inside_poly, iterate_chunks, rotation_matrix_2d


np.seterr(all='ignore')


__all__ = ['Roi', 'RectangularROI', 'CircularROI', 'PolygonalROI',
           'AbstractMplRoi', 'MplRectangularROI', 'MplCircularROI',
           'MplPolygonalROI', 'MplXRangeROI', 'MplYRangeROI',
           'XRangeROI', 'RangeROI', 'YRangeROI', 'VertexROIBase',
           'CategoricalROI', 'EllipticalROI']

PATCH_COLOR = '#FFFF00'
SCRUBBING_KEY = 'control'


def aspect_ratio(axes):
    """Returns the pixel height / width of a box that spans 1 data unit in `x` and `y`"""
    width = axes.get_position().width * axes.figure.get_figwidth()
    height = axes.get_position().height * axes.figure.get_figheight()
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    return height / width / (ymax - ymin) * (xmax - xmin)


def data_to_norm(axes, x, y):
    xy = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    pixel = axes.transData.transform(xy)
    norm = axes.transAxes.inverted().transform(pixel)
    return norm


def data_to_pixel(axes, x, y):
    xy = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    return axes.transData.transform(xy)


def pixel_to_data(axes, x, y):
    xy = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    return axes.transData.inverted().transform(xy)


def pixel_to_axes(axes, x, y):
    xy = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    return axes.transAxes.inverted().transform(xy)


class Roi(object):  # pragma: no cover

    """
    A geometrical 2D region of interest.

    Glue uses ROIs to represent user-drawn regions on plots. There
    are many specific sub-classes of Roi, but they all have a ``contains``
    method to test whether a collection of 2D points lies inside the region.
    ROI bounds are generally designed as being exclusive (that is, points
    situated exactly on a border are considered to lie outside the region).
    """

    def contains(self, x, y):
        """
        Test which of a set of (`x`, `y`) points fall within the region of interest.

        Parameters
        ----------
        x : float or array-like
            `x` coordinate(s) of point(s).
        y : float or array-like
            `y` coordinate(s) of point(s).

        Returns
        -------
        inside : bool or `~numpy.ndarray`
            An boolean iterable, where each element is `True` if the corresponding
            (`x`, `y`) tuple is inside the Roi.

        Raises
        ------
        UndefinedROI
            If not defined.
        """
        raise NotImplementedError()

    def contains3d(self, x, y, z):
        """
        Test which of a set of projected (`x`, `y`, `z`) points fall within the
        linked 2D region of interest.

        Parameters
        ----------
        x : :class:`~numpy.ndarray`
            Array of `x` locations
        y : :class:`~numpy.ndarray`
            Array of `y` locations
        z : :class:`~numpy.ndarray`
            Array of `z` locations

        Returns
        -------
        :class:`~numpy.ndarray`
            A boolean array, where each element is `True` if the corresponding
            (`x`, `y`, `z`) tuple is projected inside the associated 2D Roi.

        Raises
        ------
        UndefinedROI
            If not defined.
        """
        raise NotImplementedError()

    def center(self):
        """Return the (`x`, `y`) coordinates of the ROI center"""
        raise NotImplementedError()

    def move_to(self, x, y):
        """Translate the ROI to a center of (`x`, `y`)"""
        raise NotImplementedError()

    def defined(self):
        """Returns `True` if the ROI is defined"""
        raise NotImplementedError()

    def to_polygon(self):
        """
        Returns vertices `vx`, `vy` of a polygon approximating the Roi,
        where each is an array of vertex coordinates in `x` and `y`.
        """
        raise NotImplementedError()

    def rotate_to(self, theta):
        """
        Rotate anticlockwise around center to position angle theta.

        Parameters
        ----------
        theta : float
            Angle of anticlockwise rotation around center in radian.
        """
        raise NotImplementedError()

    def rotate_by(self, dtheta, **kwargs):
        """
        Rotate the Roi around center by angle dtheta.

        Parameters
        ----------
        dtheta : float
            Change in anticlockwise rotation angle around center in radian.
        """
        self.rotate_to(getattr(self, 'theta', 0.0) + dtheta, **kwargs)

    def copy(self):
        """Return a clone of the Roi"""
        return copy.copy(self)

    def transformed(self, xfunc=None, yfunc=None):
        """A transformed version of the Roi"""
        raise NotImplementedError()


class PointROI(Roi):

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def contains(self, x, y):
        return False

    def move_to(self, x, y):
        self.x = x
        self.y = y

    def defined(self):
        try:
            return np.isfinite([self.x, self.y]).all()
        except TypeError:
            return False

    def center(self):
        return self.x, self.y

    def reset(self):
        self.x = self.y = None


class RectangularROI(Roi):

    """
    A 2D rectangular region of interest.

    Parameters
    ----------
    xmin, xmax : float, optional
        `x` coordinates of left and right edge.
    ymin, ymax : float, optional
        `y` coordinates of lower and upper edge.
    theta : float, optional
        Angle of anticlockwise rotation around center in radian.

    Notes
    -----
        The input and ``update_limits()`` parameters specify the `x` and `y` edge
        positions *before* any rotation is applied; the size always remains
        :math:`width` = `xmax` - `xmin`; `height` = `ymax` - `ymin`.
    """

    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, theta=None):
        super(RectangularROI, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.theta = 0 if theta is None else theta

    def __str__(self):
        if self.defined() and self.theta == 0:
            return f"x=[{self.xmin:.3f}, {self.xmax:.3f}], y=[{self.ymin:.3f}, {self.ymax:.3f}]"
        elif self.defined():
            return (f"center=({self.center()[0]:.3f}, {self.center()[1]:.3f}), "
                    f"size=({self.width():.3f} x {self.height():.3f}), "
                    f"theta={self.theta:.3f} radian")
        else:
            return "Undefined Rectangular ROI"

    def center(self):
        return self.xmin + self.width() / 2, self.ymin + self.height() / 2

    def move_to(self, x, y):
        cx, cy = self.center()
        dx = x - cx
        dy = y - cy
        self.xmin += dx
        self.xmax += dx
        self.ymin += dy
        self.ymax += dy

    def rotate_to(self, theta):
        self.theta = 0 if theta is None else theta

    def transpose(self, copy=True):
        if copy:
            new = self.copy()
            new.xmin, new.xmax = self.ymin, self.ymax
            new.ymin, new.ymax = self.xmin, self.xmax
            return new

        self.xmin, self.ymin = self.ymin, self.xmin
        self.xmax, self.ymax = self.ymax, self.xmax

    def corner(self):
        return (self.xmin, self.ymin)

    def width(self):
        return self.xmax - self.xmin

    def height(self):
        return self.ymax - self.ymin

    def contains(self, x, y):
        if not self.defined():
            raise UndefinedROI

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if np.isclose(self.theta % np.pi, 0.0, atol=1e-9):
            return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax)
        elif np.isclose(self.theta % (np.pi / 2), 0.0, atol=1e-9):
            xc, yc = self.center()
            xext = self.height() / 2
            yext = self.width() / 2
            return (x > xc - xext) & (x < xc + xext) & (y > yc - yext) & (y < yc + yext)

        inside = np.zeros_like(x, dtype=bool)
        bounds = self.to_polygon()
        xc, yc = self.center()
        keep = ((x >= bounds[0].min()) & (x <= bounds[0].max()) &
                (y >= bounds[1].min()) & (y <= bounds[1].max()))
        x = x[keep] - xc
        y = y[keep] - yc
        shape = (2,) + x.shape
        x, y = (rotation_matrix_2d(-self.theta) @ [x.flatten(), y.flatten()]).reshape(shape)
        inside[keep] = (abs(x) <= self.width() / 2) & (abs(y) <= self.height() / 2)
        return inside

    def update_limits(self, xmin, ymin, xmax, ymax):
        """Update the limits (edge positions) of the rectangle before rotation"""
        self.xmin = min(xmin, xmax)
        self.xmax = max(xmin, xmax)
        self.ymin = min(ymin, ymax)
        self.ymax = max(ymin, ymax)

    def reset(self):
        """Reset the rectangular region"""
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def defined(self):
        return self.xmin is not None

    def to_polygon(self):
        """
        Returns vertices `vx`, `vy` of the rectangular region represented as a polygon,
        where each is an array of vertex coordinates in `x` and `y`.
        """
        if self.defined():
            if np.isclose(self.theta % np.pi, 0.0, atol=1e-9):
                return (np.array([self.xmin, self.xmax, self.xmax, self.xmin, self.xmin]),
                        np.array([self.ymin, self.ymin, self.ymax, self.ymax, self.ymin]))
            else:
                corners = (np.array([-1, 1, 1, -1, -1]) * self.width() / 2,
                           np.array([-1, -1, 1, 1, -1]) * self.height() / 2)
            return tuple((rotation_matrix_2d(self.theta) @ corners) +
                         np.array(self.center()).reshape((2, 1)))
        else:
            return [], []

    def transformed(self, xfunc=None, yfunc=None):
        xmin = self.xmin if xfunc is None else xfunc(self.xmin)
        xmax = self.xmax if xfunc is None else xfunc(self.xmax)
        ymin = self.ymin if yfunc is None else yfunc(self.ymin)
        ymax = self.ymax if yfunc is None else yfunc(self.ymax)
        return RectangularROI(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def __gluestate__(self, context):
        return dict(xmin=context.do(self.xmin),
                    xmax=context.do(self.xmax),
                    ymin=context.do(self.ymin),
                    ymax=context.do(self.ymax),
                    theta=context.do(self.theta))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xmin=context.object(rec['xmin']), xmax=context.object(rec['xmax']),
                   ymin=context.object(rec['ymin']), ymax=context.object(rec['ymax']),
                   theta=context.object(rec.get('theta', 0)))


class RangeROI(Roi):
    """
    A region of interest representing all points within a range in either `x` or `y`.

    Parameters
    ----------
    orientation : str
        One of 'x' or 'y', setting the axis on which to apply the range.
    min : float, optional
        Start value of the range.
    max : float, optional
        End value of the range.
    """
    def __init__(self, orientation, min=None, max=None):
        super(RangeROI, self).__init__()

        self.min = min
        self.max = max
        self.ori = orientation

    @property
    def ori(self):
        return self._ori

    @ori.setter
    def ori(self, value):
        if value in set('xy'):
            self._ori = value
        else:
            raise ValueError("Orientation must be one of 'x', 'y'")

    def __str__(self):
        if self.defined():
            return "%0.3f < %s < %0.3f" % (self.min, self.ori,
                                           self.max)
        else:
            return "Undefined %s" % type(self).__name__

    def range(self):
        return self.min, self.max

    def center(self):
        return (self.min + self.max) / 2

    def set_range(self, lo, hi):
        self.min, self.max = lo, hi

    def move_to(self, center):
        delta = center - self.center()
        self.min += delta
        self.max += delta

    def contains(self, x, y):
        if not self.defined():
            raise UndefinedROI()

        coord = x if self.ori == 'x' else y
        return (coord > self.min) & (coord < self.max)

    def transformed(self, xfunc=None, yfunc=None):
        if self.ori == 'x':
            vmin = self.min if xfunc is None else xfunc(self.min)
            vmax = self.max if xfunc is None else xfunc(self.max)
        else:
            vmin = self.min if yfunc is None else yfunc(self.min)
            vmax = self.max if yfunc is None else yfunc(self.max)
        return RangeROI(orientation=self.ori, min=vmin, max=vmax)

    def reset(self):
        self.min = None
        self.max = None

    def defined(self):
        return self.min is not None and self.max is not None

    def to_polygon(self):
        if self.defined():
            on = [self.min, self.max, self.max, self.min, self.min]
            off = [-1e100, -1e100, 1e100, 1e100, -1e100]
            x, y = (on, off) if (self.ori == 'x') else (off, on)
            return x, y
        else:
            return [], []

    def __gluestate__(self, context):
        return dict(ori=self.ori, min=context.do(self.min), max=context.do(self.max))

    @classmethod
    def __setgluestate__(cls, rec, context):
        if cls is XRangeROI or cls is YRangeROI:
            return cls(min=rec['min'], max=rec['max'])
        else:
            return cls(rec['ori'], min=rec['min'], max=rec['max'])


class XRangeROI(RangeROI):

    def __init__(self, min=None, max=None):
        super(XRangeROI, self).__init__('x', min=min, max=max)


class YRangeROI(RangeROI):

    def __init__(self, min=None, max=None):
        super(YRangeROI, self).__init__('y', min=min, max=max)


class CircularROI(Roi):

    """
    A 2D circular region of interest.

    Parameters
    ----------
    xc : float, optional
        `x` coordinate of center.
    yc : float, optional
        `y` coordinate of center.
    radius : float, optional
        Radius of the circle.
    """

    def __init__(self, xc=None, yc=None, radius=None):
        super(CircularROI, self).__init__()
        self.xc = xc
        self.yc = yc
        self.radius = radius

    def contains(self, x, y):
        if not self.defined():
            raise UndefinedROI

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        return (x - self.xc) ** 2 + (y - self.yc) ** 2 < self.radius ** 2

    def set_center(self, x, y):  # pragma: no cover
        """Set the center of the circular region"""
        warnings.warn("set_center is deprecated and may be removed "
                      "in a future release, use move_to", DeprecationWarning)
        self.move_to(x, y)

    def set_radius(self, radius):
        """Set the radius of the circular region"""
        self.radius = radius

    def get_center(self):  # pragma: no cover
        warnings.warn("get_center is deprecated and may be removed "
                      "in a future release, use center", DeprecationWarning)
        return self.center()

    def get_radius(self):
        return self.radius

    def reset(self):
        """Reset the circular region"""
        self.xc = None
        self.yc = None
        self.radius = 0.

    def defined(self):
        return (self.xc is not None and
                self.yc is not None and self.radius is not None)

    def to_polygon(self):
        if not self.defined():
            return [], []
        theta = np.linspace(0, 2 * np.pi, num=20)
        x = self.xc + self.radius * np.cos(theta)
        y = self.yc + self.radius * np.sin(theta)
        return x, y

    def transformed(self, xfunc=None, yfunc=None):
        return PolygonalROI(*self.to_polygon()).transformed(xfunc=xfunc, yfunc=yfunc)

    def center(self):
        return self.xc, self.yc

    def move_to(self, x, y):
        self.xc = x
        self.yc = y

    def __gluestate__(self, context):
        return dict(xc=context.do(self.xc),
                    yc=context.do(self.yc),
                    radius=context.do(self.radius))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xc=rec['xc'], yc=rec['yc'], radius=rec['radius'])


class EllipticalROI(Roi):
    """
    A 2D elliptical region of interest with semimajor/minor axes `radius_[xy]`.

    Parameters
    ----------
    xc : float, optional
        `x` coordinate of center.
    yc : float, optional
        `y` coordinate of center.
    radius_x : float, optional
        Semiaxis along `x` axis.
    radius_y : float, optional
        Semiaxis along `y` axis.
    theta : float, optional
        Angle of anticlockwise rotation around (`xc`, `yc`) in radian.

    Notes
    -----
    The `radius_x`, `radius_y` properties refer to the semiaxes along the `x` and `y`
    axes *before* any rotation is applied.
    """

    def __init__(self, xc=None, yc=None, radius_x=None, radius_y=None, theta=None):
        super(EllipticalROI, self).__init__()
        self.xc = xc
        self.yc = yc
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.theta = 0 if theta is None else theta

    def __str__(self):
        if self.defined():
            return (f"center=({self.xc:.3f}, {self.yc:.3f}), "
                    f"semiaxes=({self.radius_x:.3f} x {self.radius_y:.3f}), "
                    f"theta={self.theta:.3f} radian")
        else:
            return "Undefined Elliptical ROI"

    def contains(self, x, y):
        if not self.defined():
            raise UndefinedROI

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if np.isclose(self.theta % np.pi, 0.0, atol=1e-9):
            return (((x - self.xc) ** 2 / self.radius_x ** 2 +
                     (y - self.yc) ** 2 / self.radius_y ** 2) < 1.)
        elif np.isclose(self.theta % (np.pi / 2), 0.0, atol=1e-9):
            return (((x - self.xc) ** 2 / self.radius_y ** 2 +
                     (y - self.yc) ** 2 / self.radius_x ** 2) < 1.)
        else:
            # Pre-select points inside the bounding rectangle. In principle this could be
            # used to speed up the non-rotated cases above as well, but will only pay off
            # the overhead for datasets much larger than the region (e.g. < ~1 % inside).
            inside = np.zeros_like(x, dtype=bool)
            bounds = self.bounds()
            keep = ((x >= bounds[0][0]) & (x <= bounds[0][1]) &
                    (y >= bounds[1][0]) & (y <= bounds[1][1]))
            x = x[keep] - self.xc
            y = y[keep] - self.yc
            shape = (2,) + x.shape
            x, y = (rotation_matrix_2d(-self.theta) @ [x.flatten(), y.flatten()]).reshape(shape)
            inside[keep] = ((x ** 2 / self.radius_x ** 2 + y ** 2 / self.radius_y ** 2) < 1.)
            return inside

    def reset(self):
        """Reset the rectangular region"""
        self.xc = None
        self.yc = None
        self.radius_x = 0.
        self.radius_y = 0.

    def defined(self):
        return (self.xc is not None and
                self.yc is not None and
                self.radius_x is not None and
                self.radius_y is not None)

    def get_center(self):  # pragma: no cover
        warnings.warn("get_center is deprecated and may be removed "
                      "in a future release, use center", DeprecationWarning)
        return self.center()

    def to_polygon(self):
        if not self.defined():
            return [], []
        theta = np.linspace(0, 2 * np.pi, num=20)
        x = self.radius_x * np.cos(theta)
        y = self.radius_y * np.sin(theta)
        x, y = rotation_matrix_2d(self.theta) @ (x, y)
        return x + self.xc, y + self.yc

    def bounds(self):
        """Returns (conservatively estimated) boundary values in `x` and `y`"""
        if self.theta is None or np.isclose(self.theta % (np.pi), 0.0, atol=1e-9):
            return [[self.xc - self.radius_x, self.xc + self.radius_x],
                    [self.yc - self.radius_y, self.yc + self.radius_y]]
        elif np.isclose(self.theta % (np.pi / 2), 0.0, atol=1e-9):
            return [[self.xc - self.radius_y, self.xc + self.radius_y],
                    [self.yc - self.radius_x, self.yc + self.radius_x]]
        else:
            radius = max(self.radius_x, self.radius_y)
            return [[self.xc - radius, self.xc + radius],
                    [self.yc - radius, self.yc + radius]]

    def transformed(self, xfunc=None, yfunc=None):
        return PolygonalROI(*self.to_polygon()).transformed(xfunc=xfunc, yfunc=yfunc)

    def center(self):
        return self.xc, self.yc

    def move_to(self, x, y):
        self.xc = x
        self.yc = y

    def rotate_to(self, theta):
        self.theta = 0 if theta is None else theta

    def __gluestate__(self, context):
        return dict(xc=context.do(self.xc),
                    yc=context.do(self.yc),
                    radius_x=context.do(self.radius_x),
                    radius_y=context.do(self.radius_y),
                    theta=context.do(self.theta))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xc=rec['xc'], yc=rec['yc'],
                   radius_x=rec['radius_x'], radius_y=rec['radius_y'], theta=rec.get('theta', 0))


class VertexROIBase(Roi):

    """
    Class representing a set of vertices e.g. to define a 2D polygon.

    Parameters
    ----------
    vx : float or array-like, optional
        Initial `x` vertices.
    vy : float or array-like, optional
        Initial `y` vertices.

    Notes
    -----
    This class only comprises the collection of vertices, but does not
    provide a ``contains`` method.
    """

    def __init__(self, vx=None, vy=None):
        super(VertexROIBase, self).__init__()
        self.vx = [] if vx is None else list(vx)
        self.vy = [] if vy is None else list(vy)
        self.theta = 0

    def transformed(self, xfunc=None, yfunc=None):
        vx = self.vx if xfunc is None else xfunc(np.asarray(self.vx))
        vy = self.vy if yfunc is None else yfunc(np.asarray(self.vy))
        return self.__class__(vx=vx, vy=vy)

    def add_point(self, x, y):
        """
        Add another vertex to the ROI.

        Parameters
        ----------
        x : float
            The `x` coordinate of the point to add.
        y : float
            The `y` coordinate of the point to add.
        """
        self.vx.append(x)
        self.vy.append(y)

    def reset(self):
        """Reset the vertex lists and position angle"""
        self.vx = []
        self.vy = []
        self.theta = 0

    def replace_last_point(self, x, y):
        if len(self.vx) > 0:
            self.vx[-1] = x
            self.vy[-1] = y

    def remove_point(self, x, y, thresh=None):
        """
        Remove the vertex closest to a reference (`x`, `y`) point.

        Parameters
        ----------
        x : float
            The `x` coordinate of the reference point.
        y : float
            The `y` coordinate of the reference point.
        thresh : float, optional
            Threshold. If set, the vertex closest to (`x`, `y`) will only be removed
            if the distance is less than `thresh`.
        """
        if len(self.vx) == 0:
            return

        # find distance between vertices and input
        dist = [(x - a) ** 2 + (y - b) ** 2 for a, b
                in zip(self.vx, self.vy)]
        inds = range(len(dist))
        near = min(inds, key=lambda x: dist[x])

        if thresh is not None and dist[near] > (thresh ** 2):
            return

        self.vx = [self.vx[i] for i in inds if i != near]
        self.vy = [self.vy[i] for i in inds if i != near]

    def defined(self):
        return len(self.vx) > 0

    def to_polygon(self):
        return self.vx, self.vy

    def __gluestate__(self, context):
        return dict(vx=context.do(np.asarray(self.vx)),
                    vy=context.do(np.asarray(self.vy)))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(vx=context.object(rec['vx']), vy=context.object(rec['vy']))


class PolygonalROI(VertexROIBase):

    """
    A class to define 2D polygonal regions of interest.

    Parameters
    ----------
    vx : float or array-like, optional
        Initial `x` vertices.
    vy : float or array-like, optional
        Initial `y` vertices.
    """

    def __str__(self):
        result = 'Polygonal ROI ('
        result += ','.join(['(%s, %s)' % (x, y)
                            for x, y in zip(self.vx, self.vy)])
        result += ')'
        return result

    def contains(self, x, y):
        if not self.defined():
            raise UndefinedROI
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        result = points_inside_poly(x, y, np.asarray(self.vx), np.asarray(self.vy))
        return result

    # There are several possible definitions of the centre; `mean()` is
    # easiest to calculate, but not robust against adding vertices.
    def mean(self):
        """Return arithmetic mean (of vertex positions) of polygon."""

        if not self.defined():
            raise UndefinedROI
        # Do not include starting vertex twice!
        if self.vx[-1] == self.vx[0] and self.vy[:-1] == self.vy[0]:
            return np.mean(self.vx[:-1]), np.mean(self.vy[:-1])
        else:
            return np.mean(self.vx), np.mean(self.vy)

    def area(self, signed=False):
        """
        Return area of polygon using the shoelace formula.

        Parameters
        ----------
        signed : bool, optional
            If `True`, return signed area from the cross product calculation,
            indicating whether vertices are ordered clockwise (negative) or
            counter clockwise (positive).
        """

        # Use offsets to improve numerical precision
        x0, y0 = self.mean()
        x_ = self.vx - x0
        y_ = self.vy - y0

        # Shoelace formula; in case where the start vertex is not already duplicated
        # at the end, final term added manually to avoid an array copy.
        area_main = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
        if not (self.vx[-1] == self.vx[0] and self.vy[:-1] == self.vy[0]):
            area_main += x_[-1] * y_[0] - y_[-1] * x_[0]
        if signed:
            return 0.5 * area_main
        else:
            return 0.5 * np.abs(area_main)

    def centroid(self):
        """Return centroid (centre of mass) of polygon."""

        # See http://paulbourke.net/geometry/polygonmesh/
        #     https://www.ma.ic.ac.uk/~rn/centroid.pdf

        # Use vertex position offsets from mean to improve numerical precision;
        # for a triangle the mean already identifies the centroid.

        if len(self.vx) == 3:
            return self.mean()
        else:
            x0, y0 = self.mean()

        if self.vx[-1] == self.vx[0] and self.vy[:-1] == self.vy[0]:
            x_ = self.vx[:-1] - x0
            y_ = self.vy[:-1] - y0
        else:
            x_ = self.vx - x0
            y_ = self.vy - y0
        indices = np.arange(len(x_)) - 1

        xs = x_[indices] + x_
        ys = y_[indices] + y_
        dxy = x_[indices] * y_ - y_[indices] * x_
        scl = 1. / (6 * self.area(signed=True))

        return np.dot(xs, dxy) * scl + x0, np.dot(ys, dxy) * scl + y0

    def center(self):
        # centroid is more robust than mean, but
        # for linear (1D) "polygons" centroid is not defined.
        if self.area() == 0:
            return self.mean()
        else:
            return self.centroid()

    def move_to(self, new_x, new_y):
        xcen, ycen = self.center()
        xdelta = new_x - xcen
        ydelta = new_y - ycen
        self.vx = list(map(lambda x: x + xdelta, self.vx))
        self.vy = list(map(lambda y: y + ydelta, self.vy))

    def rotate_to(self, theta, center=None):
        """
        Rotate polygon to position angle `theta` around `center`.

        Parameters
        ----------
        theta : float
            Angle of anticlockwise rotation around center in radian.
        center : pair of float, optional
            Coordinates of center of rotation. Defaults to
            :meth:`~glue.core.roi.PolygonalROI.centroid`, for linear
            "polygons" to :meth:`~glue.core.roi.PolygonalROI.mean`.
        """

        theta = 0 if theta is None else theta
        center = self.center() if center is None else center
        dtheta = theta - self.theta

        if self.defined() and not np.isclose(dtheta % np.pi, 0.0, atol=1e-9):
            dx, dy = np.array([self.vx, self.vy]) - np.array(center).reshape(2, 1)
            self.vx, self.vy = (rotation_matrix_2d(dtheta) @ (dx, dy) +
                                np.array(center).reshape(2, 1)).tolist()
        self.theta = theta


class Projected3dROI(Roi):
    """
    A region of interest defined in screen coordinates.

    The screen coordinates are defined by the projection matrix.
    The projection matrix converts homogeneous coordinates (`x`, `y`, `z`, `w`),
    where `w` is implicitly 1, to homogeneous screen coordinates (usually the
    tensor product of the world coordinate vectors and the projection matrix).

    Parameters
    ----------
    2d_roi : `~glue.core.roi.Roi`, optional
        If specified, this ROI will be used in screen coordinate space.
    projection_matrix : `~numpy.ndarray`, optional
        Projection matrix defining the mapping from world onto screen coordinates.
    """

    def __init__(self, roi_2d=None, projection_matrix=None):
        super(Projected3dROI, self).__init__()
        self.roi_2d = roi_2d
        self.projection_matrix = np.asarray(projection_matrix)

    def contains3d(self, x, y, z):
        if not self.defined():
            raise UndefinedROI

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        # Since the projection can significantly increase the memory usage, we
        # do the following operation in chunks. In future we could likely use
        # e.g. vaex, dask, or other multi-threaded/fast libraries to speed this
        # and other ROI code up.

        mask = np.zeros(x.shape, dtype=bool)

        for slices in iterate_chunks(x.shape, n_max=1000000):

            # Work in homogeneous coordinates so we can support perspective
            # projections as well
            x_sub, y_sub, z_sub = x[slices], y[slices], z[slices]
            vertices = np.array([x_sub, y_sub, z_sub, np.ones(x_sub.shape)])

            # The following returns homogeneous screen coordinates
            screen_h = np.tensordot(self.projection_matrix,
                                    vertices, axes=(1, 0))

            # Convert to screen coordinates, as we don't care about z
            screen_x, screen_y = screen_h[:2] / screen_h[3]

            mask[slices] = self.roi_2d.contains(screen_x, screen_y)

        return mask

    def __gluestate__(self, context):
        return dict(roi_2d=context.id(self.roi_2d), projection_matrix=self.projection_matrix.tolist())

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(roi_2d=context.object(rec['roi_2d']), projection_matrix=np.asarray(rec['projection_matrix']))

    # TODO: these methods forward directly to roi_2d, not sure if this makes sense for all

    def contains(self, x, y):
        return self.roi_2d.contains(x, y)

    def center(self):
        return self.roi_2d.center()

    def move_to(self, x, y):
        return self.roi_2d.move_to(x, y)

    def defined(self):
        return self.roi_2d.defined()

    def to_polygon(self):
        return self.roi_2d.to_polygon()

    def transformed(self, xfunc=None, yfunc=None):
        return self.roi_2d.transformed(xfunc, yfunc)

    def rotate_to(self, theta):
        return self.roi_2d.rotate_to(theta)


class Path(VertexROIBase):

    def __str__(self):
        result = 'Path ('
        result += ','.join(['(%s, %s)' % (x, y)
                            for x, y in zip(self.vx, self.vy)])
        result += ')'
        return result


class AbstractMplRoi(object):
    """
    Base class for objects which use Matplotlib user events to edit/display ROIs.

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    roi : :class:`glue.core.roi.Roi`, optional
        If specified, this ROI will be used and updated, otherwise a new one
        will be created.
    """

    _roi_cls = None

    def __init__(self, axes, roi=None, data_space=True):
        self._axes = axes
        self._roi = roi or self._roi_cls()
        self._previous_roi = None
        self._mid_selection = False
        self._scrubbing = False
        self._background_cache = None
        self._data_space = data_space

    def _draw(self):

        # When drawing the ROI, we first keep a cache of the contents of the
        # plot then just re-plot the ROI artist on top every time for
        # performance. However, if the background cache hasn't been set, we need
        # to do a full draw.

        if self._background_cache is None or not self._axes.figure.canvas.supports_blit:
            self._axes.figure.canvas.draw_idle()
        else:
            self._axes.figure.canvas.restore_region(self._background_cache)
            self._axes.draw_artist(self._patch)
            self._axes.figure.canvas.blit()

    def roi(self):
        return self._roi.copy()

    def reset(self, include_roi=True):
        self._mid_selection = False
        self._scrubbing = False
        if include_roi:
            self._roi.reset()
        self._sync_patch()

    def active(self):
        return self._mid_selection

    def start_selection(self, event):
        raise NotImplementedError()

    def update_selection(self, event):
        raise NotImplementedError()

    def finalize_selection(self, event):
        raise NotImplementedError()

    def abort_selection(self, event):
        if self._mid_selection:
            self._restore_previous_roi()
        self.reset(include_roi=False)

    def _sync_patch(self):
        raise NotImplementedError()

    def _store_previous_roi(self):
        self._previous_roi = self._roi.copy()

    def _store_background(self):
        self._background_cache = self._axes.figure.canvas.copy_from_bbox(self._axes.bbox)

    def _reset_background(self):

        # The purpose of this method is to provide a way to reset the background
        # when the figure is changed (e.g. while panning/zooming) while the ROI
        # is in the middle of being plotted (this is relevant for 'persistent'
        # ROIs such as path selections or lasso selections).

        if self._patch is None or not self._patch.get_visible():
            return

        self._background_cache = None
        self._patch.set_visible(False)
        self._axes.figure.canvas.draw()
        self._store_background()
        self._patch.set_visible(True)
        self._axes.figure.canvas.draw_idle()

    def _restore_previous_roi(self):
        self._roi = self._previous_roi


class MplPickROI(AbstractMplRoi):
    """
    Matplotlib ROI for point selections

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    """

    _roi_cls = PointROI

    def _draw(self):
        pass

    def start_selection(self, event):
        self._roi.x = event.xdata
        self._roi.y = event.ydata

    def update_selection(self, event):
        self._roi.x = event.xdata
        self._roi.y = event.ydata

    def finalize_selection(self, event):
        self._roi.x = event.xdata
        self._roi.y = event.ydata

    def _sync_patch(self):
        pass


class MplRectangularROI(AbstractMplRoi):
    """
    Matplotlib ROI for rectangular selections

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    """

    _roi_cls = RectangularROI

    def __init__(self, axes, data_space=True):

        super(MplRectangularROI, self).__init__(axes, data_space=data_space)

        self._xi = None
        self._yi = None
        self.plot_opts = {'edgecolor': PATCH_COLOR,
                          'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._patch = Rectangle((0., 0.), 1., 1., zorder=100)
        self._patch.set_visible(False)
        if not self._data_space:
            self._patch.set_transform(self._axes.transAxes)

    def start_selection(self, event):
        if event.inaxes != self._axes:
            return False

        if self._data_space:
            xval = event.xdata
            yval = event.ydata
        else:
            axes_trans = self._axes.transAxes.inverted()
            xval, yval = axes_trans.transform([event.x, event.y])

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False
            elif not self._roi.contains(xval, yval):
                return False

        self._store_previous_roi()
        self._store_background()

        self._xi = xval
        self._yi = yval

        if event.key == SCRUBBING_KEY:
            self._scrubbing = True
            self._cx, self._cy = self._roi.center()
        else:
            self.reset()
            self._roi.update_limits(self._xi, self._xi,
                                    self._yi, self._yi)

        self._mid_selection = True

        self._sync_patch()
        self._draw()

    def update_selection(self, event):

        if not self._mid_selection or event.inaxes != self._axes:
            return False

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False

        if self._data_space:
            xval = event.xdata
            yval = event.ydata
        else:
            axes_trans = self._axes.transAxes.inverted()
            xval, yval = axes_trans.transform([event.x, event.y])

        if self._scrubbing:
            self._roi.move_to(self._cx + xval - self._xi,
                              self._cy + yval - self._yi)
        else:
            self._roi.update_limits(min(xval, self._xi),
                                    min(yval, self._yi),
                                    max(xval, self._xi),
                                    max(yval, self._yi))

        self._sync_patch()
        self._draw()

    def finalize_selection(self, event):
        self._scrubbing = False
        self._mid_selection = False
        self._patch.set_visible(False)
        self._draw()

    def _sync_patch(self):
        if self._roi.defined():
            corner = self._roi.corner()
            width = self._roi.width()
            height = self._roi.height()
            self._patch.set_xy(corner)
            self._patch.set_width(width)
            self._patch.set_height(height)
            self._patch.set(**self.plot_opts)
            self._patch.set_visible(True)
            self._axes.add_patch(self._patch)
        else:
            if self._patch in self._axes.patches:
                self._patch._remove_method(self._patch)
            self._patch.set_visible(False)

    def __str__(self):
        return "MPL Rectangle: %s" % self._patch


class MplXRangeROI(AbstractMplRoi):
    """
    Matplotlib ROI for x range selections

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    """

    _roi_cls = XRangeROI

    def __init__(self, axes, data_space=True):

        super(MplXRangeROI, self).__init__(axes, data_space=data_space)

        self._xi = None

        self.plot_opts = {'edgecolor': PATCH_COLOR,
                          'facecolor': PATCH_COLOR,
                          'alpha': 0.3}
        if self._data_space:
            trans = blended_transform_factory(self._axes.transData,
                                              self._axes.transAxes)
        else:
            trans = self._axes.transAxes
        self._patch = Rectangle((0., 0.), 1., 1., transform=trans, zorder=100)
        self._patch.set_visible(False)

    def start_selection(self, event):

        if event.inaxes != self._axes:
            return False
        if self._data_space:
            x_val = event.xdata
            y_val = event.ydata
        else:
            transform = self._axes.transAxes.inverted()
            x_val, y_val = transform.transform([event.x, event.y])
        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False
            elif not self._roi.contains(x_val, y_val):
                return False

        self._store_previous_roi()
        self._store_background()

        if event.key == SCRUBBING_KEY:
            self._scrubbing = True
            self._dx = x_val - self._roi.center()
        else:
            self.reset()
            self._roi.set_range(x_val, x_val)
            self._xi = x_val

        self._mid_selection = True

        self._sync_patch()
        self._draw()

    def update_selection(self, event):

        if not self._mid_selection or event.inaxes != self._axes:
            return False

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False

        if self._data_space:
            xval = event.xdata
        else:
            axes_trans = self._axes.transAxes.inverted()
            xval, _ = axes_trans.transform([event.x, event.y])

        if self._scrubbing:
            self._roi.move_to(xval + self._dx)
        else:
            self._roi.set_range(min(xval, self._xi),
                                max(xval, self._xi))

        self._sync_patch()
        self._draw()

    def finalize_selection(self, event):
        self._scrubbing = False
        self._mid_selection = False
        self._patch.set_visible(False)
        self._draw()

    def _sync_patch(self):
        if self._roi.defined():
            rng = self._roi.range()
            self._patch.set_xy((rng[0], 0))
            self._patch.set_width(rng[1] - rng[0])
            self._patch.set_height(1)
            self._patch.set(**self.plot_opts)
            self._patch.set_visible(True)
            self._axes.add_patch(self._patch)
        else:
            if self._patch in self._axes.patches:
                self._patch._remove_method(self._patch)
            self._patch.set_visible(False)


class MplYRangeROI(AbstractMplRoi):
    """
    Matplotlib ROI for y range selections

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    """

    _roi_cls = YRangeROI

    def __init__(self, axes, data_space=True):

        super(MplYRangeROI, self).__init__(axes, data_space=data_space)

        self._yi = None

        self.plot_opts = {'edgecolor': PATCH_COLOR,
                          'facecolor': PATCH_COLOR,
                          'alpha': 0.3}
        if self._data_space:
            trans = blended_transform_factory(self._axes.transAxes,
                                              self._axes.transData)
        else:
            trans = self._axes.transAxes
        self._patch = Rectangle((0., 0.), 1., 1., transform=trans, zorder=100)
        self._patch.set_visible(False)

    def start_selection(self, event):

        if event.inaxes != self._axes:
            return False

        if self._data_space:
            xval = event.xdata
            yval = event.ydata
        else:
            axes_trans = self._axes.transAxes.inverted()
            xval, yval = axes_trans.transform([event.x, event.y])

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False
            elif not self._roi.contains(xval, yval):
                return False

        self._store_previous_roi()
        self._store_background()

        if event.key == SCRUBBING_KEY:
            self._scrubbing = True
            self._dy = yval - self._roi.center()
        else:
            self.reset()
            self._roi.set_range(yval, yval)
            self._yi = yval

        self._mid_selection = True

        self._sync_patch()
        self._draw()

    def update_selection(self, event):

        if not self._mid_selection or event.inaxes != self._axes:
            return False

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False

        if self._data_space:
            yval = event.ydata
        else:
            axes_trans = self._axes.transAxes.inverted()
            _, yval = axes_trans.transform([event.x, event.y])

        if self._scrubbing:
            self._roi.move_to(yval + self._dy)
        else:
            self._roi.set_range(min(yval, self._yi),
                                max(yval, self._yi))

        self._sync_patch()
        self._draw()

    def finalize_selection(self, event):
        self._scrubbing = False
        self._mid_selection = False
        self._patch.set_visible(False)
        self._draw()

    def _sync_patch(self):
        if self._roi.defined():
            rng = self._roi.range()
            self._patch.set_xy((0, rng[0]))
            self._patch.set_height(rng[1] - rng[0])
            self._patch.set_width(1)
            self._patch.set(**self.plot_opts)
            self._patch.set_visible(True)
            self._axes.add_patch(self._patch)
        else:
            if self._patch in self._axes.patches:
                self._patch._remove_method(self._patch)
            self._patch.set_visible(False)


class MplCircularROI(AbstractMplRoi):
    """
    Matplotlib ROI for circular selections

    Since circles on the screen may not be circles in the data (due, e.g., to
    logarithmic scalings on the axes), the ultimate ROI that is created is a
    polygonal ROI

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    """

    _roi_cls = CircularROI

    def __init__(self, axes, data_space=True):

        super(MplCircularROI, self).__init__(axes, data_space=data_space)

        self.plot_opts = {'edgecolor': PATCH_COLOR,
                          'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._xi = None
        self._yi = None

        self._patch = Ellipse((0., 0.), transform=IdentityTransform(),
                              width=0., height=0., zorder=100)
        self._patch.set_visible(False)

    def _sync_patch(self):
        if self._roi.defined():
            xy = self._roi.center()
            r = self._roi.get_radius()
            self._patch.center = xy
            self._patch.width = 2. * r
            self._patch.height = 2. * r
            self._patch.set(**self.plot_opts)
            self._patch.set_visible(True)
            self._axes.add_patch(self._patch)
        else:
            if self._patch in self._axes.patches:
                self._patch._remove_method(self._patch)
            self._patch.set_visible(False)

    def start_selection(self, event):

        if event.inaxes != self._axes:
            return False

        xy = data_to_pixel(self._axes, [event.xdata], [event.ydata])
        xi = xy[0, 0]
        yi = xy[0, 1]

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False
            elif not self._roi.contains(xi, yi):
                return False

        self._store_previous_roi()
        self._store_background()

        if event.key == SCRUBBING_KEY:
            self._scrubbing = True
            (xc, yc) = self._roi.center()
            self._dx = xc - xi
            self._dy = yc - yi
        else:
            self.reset()
            self._roi.move_to(xi, yi)
            self._roi.set_radius(0.)
            self._xi = xi
            self._yi = yi

        self._mid_selection = True

        self._sync_patch()
        self._draw()

    def update_selection(self, event):

        if not self._mid_selection or event.inaxes != self._axes:
            return False

        xy = data_to_pixel(self._axes, [event.xdata], [event.ydata])
        xi = xy[0, 0]
        yi = xy[0, 1]

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False

        if self._scrubbing:
            self._roi.move_to(xi + self._dx, yi + self._dy)
        else:
            dx = xy[0, 0] - self._xi
            dy = xy[0, 1] - self._yi
            self._roi.set_radius(np.hypot(dx, dy))

        self._sync_patch()
        self._draw()

    def roi(self):
        if not self._roi.defined():
            return PolygonalROI()

        # Get the circular ROI parameters in pixel units
        xy_center = self._roi.center()
        rad = self._roi.get_radius()

        # At this point, if one of the axes is not linear, we convert to a polygon
        if (self._axes.get_xscale() != 'linear' or self._axes.get_yscale() != 'linear') and self._data_space:
            theta = np.linspace(0, 2 * np.pi, num=200)
            x = xy_center[0] + rad * np.cos(theta)
            y = xy_center[1] + rad * np.sin(theta)
            xy_data = pixel_to_data(self._axes, x, y)
            vx = xy_data[:, 0].ravel().tolist()
            vy = xy_data[:, 1].ravel().tolist()
            result = PolygonalROI(vx, vy)
        else:
            # We should now check if the radius in data coordinates is the same
            # along x and y, as if so then we can return a circle, otherwise we
            # should return an ellipse.
            x = xy_center[0] + np.array([0, 0, rad])
            y = xy_center[1] + np.array([0, rad, rad])
            xy_data = pixel_to_data(self._axes, x, y) if self._data_space else pixel_to_axes(self._axes, x, y)
            rx = xy_data[2, 0] - xy_data[0, 0]
            ry = xy_data[1, 1] - xy_data[0, 1]
            xc, yc = xy_data[0, :]
            if np.allclose(rx, ry):
                return CircularROI(xc=xc, yc=yc, radius=rx)
            else:
                return EllipticalROI(xc=xc, yc=yc, radius_x=rx, radius_y=ry)

        return result

    def finalize_selection(self, event):
        self._scrubbing = False
        self._mid_selection = False
        self._patch.set_visible(False)
        self._draw()


class MplPolygonalROI(AbstractMplRoi):
    """
    Matplotlib ROI for polygon selections

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    roi : :class:`glue.core.roi.Roi`, optional
        If specified, this ROI will be used and updated, otherwise a new one
        will be created.
    """

    _roi_cls = PolygonalROI

    def __init__(self, axes, roi=None, data_space=True):

        super(MplPolygonalROI, self).__init__(axes, roi=roi, data_space=data_space)

        self.plot_opts = {'edgecolor': PATCH_COLOR,
                          'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._patch = Polygon(np.array(list(zip([0, 1], [0, 1]))), zorder=100)
        self._patch.set_visible(False)
        if not self._data_space:
            self._patch.set_transform(self._axes.transAxes)

    def _sync_patch(self):
        if self._roi.defined():
            x, y = self._roi.to_polygon()
            self._patch.set_xy(list(zip(x + [x[0]], y + [y[0]])))
            self._patch.set_visible(True)
            self._patch.set(**self.plot_opts)
            self._axes.add_patch(self._patch)
        else:
            if self._patch in self._axes.patches:
                self._patch._remove_method(self._patch)
            self._patch.set_visible(False)

    def start_selection(self, event, scrubbing=False):

        if event.inaxes != self._axes:
            return False

        if self._data_space:
            xval = event.xdata
            yval = event.ydata
        else:
            axes_trans = self._axes.transAxes.inverted()
            xval, yval = axes_trans.transform([event.x, event.y])

        if scrubbing or event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False
            elif not self._roi.contains(xval, yval):
                return False

        self._store_previous_roi()
        self._store_background()

        if scrubbing or event.key == SCRUBBING_KEY:
            self._scrubbing = True
            self._cx = xval
            self._cy = yval
        else:
            self.reset()
            self._roi.add_point(xval, yval)

        self._mid_selection = True

        self._sync_patch()
        self._draw()

    def update_selection(self, event):

        if not self._mid_selection or event.inaxes != self._axes:
            return False

        if event.key == SCRUBBING_KEY:
            if not self._roi.defined():
                return False

        if self._data_space:
            xval = event.xdata
            yval = event.ydata
        else:
            axes_trans = self._axes.transAxes.inverted()
            xval, yval = axes_trans.transform([event.x, event.y])

        if self._scrubbing:
            old_x, old_y = self._roi.center()
            new_x = old_x + xval - self._cx
            new_y = old_y + yval - self._cy
            self._roi.move_to(new_x, new_y)
            self._cx = xval
            self._cy = yval
        else:
            self._roi.add_point(xval, yval)

        self._sync_patch()
        self._draw()

    def finalize_selection(self, event):
        self._scrubbing = False
        self._mid_selection = False
        self._patch.set_visible(False)
        self._draw()


class MplPathROI(MplPolygonalROI):
    """
    Matplotlib ROI for path selections

    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    """

    _roi_cls = Path

    def __init__(self, axes, roi=None):

        super(MplPolygonalROI, self).__init__(axes)

        self.plot_opts = {'edgecolor': PATCH_COLOR,
                          'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._patch = None

    def start_selection(self, event):

        if self._patch is not None:
            self._patch.remove()
            self._patch = None

        self._background_cache = None
        self._axes.figure.canvas.draw()

        super(MplPathROI, self).start_selection(event)

    def _sync_patch(self):

        if self._patch is not None:
            self._patch.remove()
            self._patch = None

        if self._roi.defined():
            x, y = self._roi.to_polygon()
            p = MplPath(np.column_stack((x, y)))
            self._patch = PathPatch(p, transform=self._axes.transData)
            self._patch.set_visible(True)
            self._patch.set(**self.plot_opts)
            self._axes.add_artist(self._patch)

    def finalize_selection(self, event):
        self._mid_selection = False
        if self._patch is not None:
            self._patch.remove()
            self._patch = None
        self._draw()


class CategoricalROI(Roi):
    """
    A ROI abstraction to represent selections of categorical data.
    """

    def __init__(self, categories=None):
        if categories is None:
            self.categories = None
        else:
            self.update_categories(categories)

    def to_polygon(self):
        """ Just not possible.
        """
        raise NotImplementedError

    def _categorical_helper(self, indata):
        """
        A helper function to do the rigamaroll of getting categorical data.

        Parameters
        ----------
        indata : object
            Any type of input data

        Returns
        -------
            The best guess at the categorical data associated with indata.
        """

        try:
            if isinstance(indata, CategoricalComponent):
                return indata.data
            else:
                return indata[:]
        except AttributeError:
            return np.asarray(indata)

    def contains(self, x, y):
        """
        Test whether a set categorical elements fall within the region of interest.

        Parameters
        ----------
        x : array-like
            An array-like object of categories (includes `CategoricalComponents`).
        y : object or None
            Unused but required for compatibility

        Returns
        -------
           A list of True/False values, for whether each `x` value falls
           within the ROI

        """
        if self.categories is None or len(self.categories) == 0:
            return np.zeros(x.shape, dtype=bool)
        else:
            check = self._categorical_helper(x)
            index = np.minimum(np.searchsorted(self.categories, check),
                               len(self.categories) - 1)
            return self.categories[index] == check

    def update_categories(self, categories):
        self.categories = np.unique(self._categorical_helper(categories))

    def defined(self):
        return self.categories is not None

    def reset(self):
        self.categories = None

    @staticmethod
    def from_range(categories, lo, hi):
        """
        Utility function to help construct the ROI from a range.

        Parameters
        ----------
        categories : object
            Anything understood by ``._categorical_helper`` ... array, list or component.
        lo : int or float
            Lower bound of the range (rounded up to next integer)
        hi : int or float
            Upper bound of the range (rounded up to next integer)

        Returns
        -------
            `CategoricalROI` object
        """

        # Convert lo and hi to integers. Note that if lo or hi are negative,
        # which can happen if the user zoomed out, we need to reset the to zero
        # otherwise they will have strange effects when slicing the categories.

        # Note that we used ceil for lo, because if lo is 0.9 then we should
        # only select 1 and above.

        lo = np.intp(np.ceil(lo) if lo > 0 else 0)
        hi = np.intp(np.ceil(hi) if hi > 0 else 0)

        roi = CategoricalROI()
        roi.update_categories(categories[lo:hi])

        return roi

    def __gluestate__(self, context):
        return dict(categories=self.categories.tolist())

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(categories=rec['categories'])
