import copy

import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle, Path as MplPath, PathPatch
from matplotlib.transforms import IdentityTransform, blended_transform_factory

from glue.core.component import CategoricalComponent
from glue.core.exceptions import UndefinedROI
from glue.utils import points_inside_poly, iterate_chunks


np.seterr(all='ignore')


__all__ = ['Roi', 'RectangularROI', 'CircularROI', 'PolygonalROI',
           'AbstractMplRoi', 'MplRectangularROI', 'MplCircularROI',
           'MplPolygonalROI', 'MplXRangeROI', 'MplYRangeROI',
           'XRangeROI', 'RangeROI', 'YRangeROI', 'VertexROIBase',
           'CategoricalROI', 'EllipticalROI']

PATCH_COLOR = '#FFFF00'
SCRUBBING_KEY = 'control'


def aspect_ratio(axes):
    """ Returns the pixel height / width of a box that spans 1
    data unit in x and y
    """
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

    Glue uses Roi's to represent user-drawn regions on plots. There
    are many specific sub-classes of Roi, but they all have a ``contains``
    method to test whether a collection of 2D points lies inside the region.
    """

    def contains(self, x, y):
        """Return true/false for each x/y pair.

        :param x: Array of X locations
        :param y: Array of Y locations

        :returns: A Boolean array, where each element is True
                  if the corresponding (x,y) tuple is inside the Roi.

        :raises: UndefinedROI exception if not defined
        """
        raise NotImplementedError()

    def contains3d(self, x, y, z):
        """Return true/false for each x/y/z pair.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Array of x locations
        y : :class:`numpy.ndarray`
            Array of y locations
        z : :class:`numpy.ndarray`
            Array of z locations

        Returns
        -------
        :class:`numpy.ndarray`
            A boolean array, where each element is `True` if the corresponding
            (x,y,z) tuple is inside the Roi.

        Raises
        ------
        UndefinedROI
            if not defined
        """
        raise NotImplementedError()

    def center(self):
        """Return the (x,y) coordinates of the ROI center"""
        raise NotImplementedError()

    def move_to(self, x, y):
        """Translate the ROI to a center of (x, y)"""
        raise NotImplementedError()

    def defined(self):
        """ Returns whether or not the subset is properly defined """
        raise NotImplementedError()

    def to_polygon(self):
        """ Returns a tuple of x and y points, approximating the ROI
        as a polygon."""
        raise NotImplementedError()

    def copy(self):
        """
        Return a clone of the ROI
        """
        return copy.copy(self)

    def transformed(self, xfunc=None, yfunc=None):
        """
        A transformed version of the ROI
        """
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
    """

    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None):
        super(RectangularROI, self).__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __str__(self):
        if self.defined():
            return "x=[%0.3f, %0.3f], y=[%0.3f, %0.3f]" % (self.xmin,
                                                           self.xmax,
                                                           self.ymin,
                                                           self.ymax)
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
        """
        Test whether a set of (x,y) points falls within
        the region of interest

        :param x: A scalar or numpy array of x points
        :param y: A scalar or numpy array of y points

        *Returns*

            A list of True/False values, for whether each (x,y)
            point falls within the ROI
        """

        if not self.defined():
            raise UndefinedROI

        return (x > self.xmin) & (x < self.xmax) & \
               (y > self.ymin) & (y < self.ymax)

    def update_limits(self, xmin, ymin, xmax, ymax):
        """
        Update the limits of the rectangle
        """
        self.xmin = min(xmin, xmax)
        self.xmax = max(xmin, xmax)
        self.ymin = min(ymin, ymax)
        self.ymax = max(ymin, ymax)

    def reset(self):
        """
        Reset the rectangular region.
        """
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def defined(self):
        return self.xmin is not None

    def to_polygon(self):
        if self.defined():
            return (np.array([self.xmin, self.xmax, self.xmax, self.xmin, self.xmin]),
                    np.array([self.ymin, self.ymin, self.ymax, self.ymax, self.ymin]))
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
                    ymax=context.do(self.ymax))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xmin=context.object(rec['xmin']), xmax=context.object(rec['xmax']),
                   ymin=context.object(rec['ymin']), ymax=context.object(rec['ymax']))


class RangeROI(Roi):

    def __init__(self, orientation, min=None, max=None):
        """:param orientation: 'x' or 'y'. Sets which axis to range"""
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
    """

    def __init__(self, xc=None, yc=None, radius=None):
        super(CircularROI, self).__init__()
        self.xc = xc
        self.yc = yc
        self.radius = radius

    def contains(self, x, y):
        """
        Test whether a set of (x,y) points falls within
        the region of interest

        :param x: A list of x points
        :param y: A list of y points

        *Returns*

           A list of True/False values, for whether each (x,y)
           point falls within the ROI

        """
        if not self.defined():
            raise UndefinedROI

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        return (x - self.xc) ** 2 + (y - self.yc) ** 2 < self.radius ** 2

    def set_center(self, x, y):
        """
        Set the center of the circular region
        """
        self.xc = x
        self.yc = y

    def set_radius(self, radius):
        """
        Set the radius of the circular region
        """
        self.radius = radius

    def get_center(self):
        return self.xc, self.yc

    def get_radius(self):
        return self.radius

    def reset(self):
        """
        Reset the rectangular region.
        """
        self.xc = None
        self.yc = None
        self.radius = 0.

    def defined(self):
        """ Returns True if the ROI is defined """
        return self.xc is not None and \
            self.yc is not None and self.radius is not None

    def to_polygon(self):
        """ Returns x, y, where each is a list of points """
        if not self.defined():
            return [], []
        theta = np.linspace(0, 2 * np.pi, num=20)
        x = self.xc + self.radius * np.cos(theta)
        y = self.yc + self.radius * np.sin(theta)
        return x, y

    def transformed(self, xfunc=None, yfunc=None):
        return PolygonalROI(*self.to_polygon()).transformed(xfunc=xfunc, yfunc=yfunc)

    def move_to(self, xdelta, ydelta):
        self.xc += xdelta
        self.yc += ydelta

    def __gluestate__(self, context):
        return dict(xc=context.do(self.xc),
                    yc=context.do(self.yc),
                    radius=context.do(self.radius))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xc=rec['xc'], yc=rec['yc'], radius=rec['radius'])


class EllipticalROI(Roi):
    """
    A 2D elliptical region of interest.
    """

    def __init__(self, xc=None, yc=None, radius_x=None, radius_y=None, theta=None):
        super(EllipticalROI, self).__init__()
        if theta is not None:
            raise NotImplementedError("Rotated ellipses are not yet supported")
        self.xc = xc
        self.yc = yc
        self.radius_x = radius_x
        self.radius_y = radius_y

    def contains(self, x, y):
        """
        Test whether a set of (x,y) points falls within
        the region of interest

        :param x: A list of x points
        :param y: A list of y points

        *Returns*

           A list of True/False values, for whether each (x,y)
           point falls within the ROI

        """
        if not self.defined():
            raise UndefinedROI

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        return (((x - self.xc) ** 2 / self.radius_x ** 2 +
                 (y - self.yc) ** 2 / self.radius_y ** 2) < 1.)

    def reset(self):
        """
        Reset the rectangular region.
        """
        self.xc = None
        self.yc = None
        self.radius_x = 0.
        self.radius_y = 0.

    def defined(self):
        """ Returns True if the ROI is defined """
        return (self.xc is not None and
                self.yc is not None and
                self.radius_x is not None and
                self.radius_y is not None)

    def to_polygon(self):
        """ Returns x, y, where each is a list of points """
        if not self.defined():
            return [], []
        theta = np.linspace(0, 2 * np.pi, num=20)
        x = self.xc + self.radius_x * np.cos(theta)
        y = self.yc + self.radius_y * np.sin(theta)
        return x, y

    def transformed(self, xfunc=None, yfunc=None):
        return PolygonalROI(*self.to_polygon()).transformed(xfunc=xfunc, yfunc=yfunc)

    def move_to(self, xdelta, ydelta):
        self.xc += xdelta
        self.yc += ydelta

    def __gluestate__(self, context):
        return dict(xc=context.do(self.xc),
                    yc=context.do(self.yc),
                    radius_x=context.do(self.radius_x),
                    radius_y=context.do(self.radius_y))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xc=rec['xc'], yc=rec['yc'],
                   radius_x=rec['radius_x'], radius_y=rec['radius_y'])


class VertexROIBase(Roi):

    def __init__(self, vx=None, vy=None):
        """
        :param vx: initial x vertices
        :type vx: list
        :param vy: initial y vertices
        :type vy: list
        """
        super(VertexROIBase, self).__init__()
        self.vx = vx
        self.vy = vy
        if self.vx is None:
            self.vx = []
        if self.vy is None:
            self.vy = []

    def transformed(self, xfunc=None, yfunc=None):
        vx = self.vx if xfunc is None else xfunc(np.asarray(self.vx))
        vy = self.vy if yfunc is None else yfunc(np.asarray(self.vy))
        return self.__class__(vx=vx, vy=vy)

    def add_point(self, x, y):
        """
        Add another vertex to the ROI

        :param x: The x coordinate
        :param y: The y coordinate
        """
        self.vx.append(x)
        self.vy.append(y)

    def reset(self):
        """
        Reset the vertex list.
        """
        self.vx = []
        self.vy = []

    def replace_last_point(self, x, y):
        if len(self.vx) > 0:
            self.vx[-1] = x
            self.vy[-1] = y

    def remove_point(self, x, y, thresh=None):
        """Remove the vertex closest to a reference (xy) point

        :param x: The x coordinate of the reference point
        :param y: The y coordinate of the reference point

        :param thresh: An optional threshold. If present, the vertex
                closest to (x,y) will only be removed if the distance
                is less than thresh

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
        return cls(vx=context.object(rec['vx']),
                   vy=context.object(rec['vy']))


class PolygonalROI(VertexROIBase):

    """
    A class to define 2D polygonal regions-of-interest
    """

    def __str__(self):
        result = 'Polygonal ROI ('
        result += ','.join(['(%s, %s)' % (x, y)
                            for x, y in zip(self.vx, self.vy)])
        result += ')'
        return result

    def contains(self, x, y):
        """
        Test whether a set of (x,y) points falls within
        the region of interest

        :param x: A list of x points
        :param y: A list of y points

        *Returns*

           A list of True/False values, for whether each (x,y)
           point falls within the ROI

        """
        if not self.defined():
            raise UndefinedROI
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        result = points_inside_poly(x, y, self.vx, self.vy)
        return result

    def move_to(self, xdelta, ydelta):
        self.vx = list(map(lambda x: x + xdelta, self.vx))
        self.vy = list(map(lambda y: y + ydelta, self.vy))


class Projected3dROI(Roi):
    """"A region of interest defined in screen coordinates.

    The screen coordinates are defined by the projection matrix.
    The projection matrix converts homogeneous coordinates (x, y, z, w), where
    w is implicitly 1, to homogeneous screen coordinates (usually the product
    of the world and projection matrix).
    """

    def __init__(self, roi_2d=None, projection_matrix=None):
        super(Projected3dROI, self).__init__()
        self.roi_2d = roi_2d
        self.projection_matrix = np.asarray(projection_matrix)

    def contains3d(self, x, y, z):
        """
        Test whether the projected coordinates are contained in the 2d ROI.
        """

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
    axes : `~matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    roi : `~glue.core.roi.Roi`, optional
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
    axes : `~matplotlib.axes.Axes`
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
    axes : `~matplotlib.axes.Axes`
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
        self._axes.add_patch(self._patch)

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
        else:
            self._patch.set_visible(False)

    def __str__(self):
        return "MPL Rectangle: %s" % self._patch


class MplXRangeROI(AbstractMplRoi):
    """
    Matplotlib ROI for x range selections

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
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
        self._axes.add_patch(self._patch)

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
        else:
            self._patch.set_visible(False)


class MplYRangeROI(AbstractMplRoi):
    """
    Matplotlib ROI for y range selections

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
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
        self._axes.add_patch(self._patch)

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
        else:
            self._patch.set_visible(False)


class MplCircularROI(AbstractMplRoi):
    """
    Matplotlib ROI for circular selections

    Since circles on the screen may not be circles in the data (due, e.g., to
    logarithmic scalings on the axes), the ultimate ROI that is created is a
    polygonal ROI

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
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
        self._axes.add_patch(self._patch)

    def _sync_patch(self):
        if self._roi.defined():
            xy = self._roi.get_center()
            r = self._roi.get_radius()
            self._patch.center = xy
            self._patch.width = 2. * r
            self._patch.height = 2. * r
            self._patch.set(**self.plot_opts)
            self._patch.set_visible(True)
        else:
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
            (xc, yc) = self._roi.get_center()
            self._dx = xc - xi
            self._dy = yc - yi
        else:
            self.reset()
            self._roi.set_center(xi, yi)
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
            self._roi.set_center(xi + self._dx, yi + self._dy)
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
        xy_center = self._roi.get_center()
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
    axes : `~matplotlib.axes.Axes`
        The Matplotlib axes to draw to.
    roi : `~glue.core.roi.Roi`, optional
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
        self._axes.add_patch(self._patch)

    def _sync_patch(self):
        if self._roi.defined():
            x, y = self._roi.to_polygon()
            self._patch.set_xy(list(zip(x + [x[0]],
                                        y + [y[0]])))
            self._patch.set_visible(True)
            self._patch.set(**self.plot_opts)
        else:
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
            self._roi.move_to(xval - self._cx,
                              yval - self._cy)
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
    axes : `~matplotlib.axes.Axes`
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

        :param indata: Any type of input data
        :return: The best guess at the categorical data associated with indata
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
        Test whether a set categorical elements fall within
        the region of interest

        :param x: Any array-like object of categories
                 (includes CategoricalComponenets)
        :param y: Unused but required for compatibility

        *Returns*

           A list of True/False values, for whether each x value falls
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
        """ Returns True if the ROI is defined """
        return self.categories is not None

    def reset(self):
        self.categories = None

    @staticmethod
    def from_range(categories, lo, hi):
        """
        Utility function to help construct the Roi from a range.

        :param cat_comp: Anything understood by ._categorical_helper ... array, list or component
        :param lo: lower bound of the range
        :param hi: upper bound of the range
        :return: CategoricalROI object
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
