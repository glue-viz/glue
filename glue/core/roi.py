from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import datetime
from matplotlib.patches import Polygon, Rectangle, Ellipse, PathPatch
from matplotlib.patches import Path as mplPath
from matplotlib.transforms import IdentityTransform, blended_transform_factory
import matplotlib.dates as mdate
import copy

np.seterr(all='ignore')

from .exceptions import UndefinedROI

__all__ = ['Roi', 'RectangularROI', 'CircularROI', 'PolygonalROI',
           'AbstractMplRoi', 'MplRectangularROI', 'MplCircularROI',
           'MplPolygonalROI', 'MplXRangeROI', 'MplYRangeROI',
           'XRangeROI', 'RangeROI', 'YRangeROI','VertexROIBase']

PATCH_COLOR = '#FFFF00'


try:
    from matplotlib.nxutils import points_inside_poly
except ImportError:  # nxutils removed in MPL v1.3
    from matplotlib.path import Path as mplPath

    def points_inside_poly(xypts, xyvts):
        p = mplPath(xyvts)
        return p.contains_points(xypts)


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


class Roi(object):  # pragma: no cover

    """
    A geometrical 2D region of interest.

    Glue uses Roi's to represent user-drawn regions on plots. There
    are many specific subtypes of Roi, but they all have a ``contains``
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
        raise NotImplementedError

    def copy(self):
        """
        Return a clone of the ROI
        """
        return copy.copy(self)


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
            return [self.xmin, self.xmax, self.xmax, self.xmin, self.xmin], \
                [self.ymin, self.ymin, self.ymax, self.ymax, self.ymin]
        else:
            return [], []

    def __gluestate__(self, context):
        return dict(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xmin=rec['xmin'], xmax=rec['xmax'],
                   ymin=rec['ymin'], ymax=rec['ymax'])


class RangeROI(Roi):

    def __init__(self, orientation, min=None, max=None):
        """:param orientation: 'x' or 'y'. Sets which axis to range"""
        super(RangeROI, self).__init__()
        if orientation not in ['x', 'y']:
            raise TypeError("Orientation must be one of 'x', 'y'")

        self.min = min
        self.max = max
        self.ori = orientation

    def __str__(self):
        if self.defined():
            return "%0.3f < %s < %0.3f" % (self.min, self.ori,
                                           self.max)
        else:
            return "Undefined %s" % type(self).__name__

    def range(self):
        return self.min, self.max

    def set_range(self, lo, hi):
        self.min, self.max = lo, hi

    def contains(self, x, y):
        if not self.defined():
            raise UndefinedROI()

        coord = x if self.ori == 'x' else y
        return (coord > self.min) & (coord < self.max)

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
        return dict(ori=self.ori, min=self.min, max=self.max)

    @classmethod
    def __setgluestate__(cls, rec, context):
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

    def __gluestate__(self, context):
        return dict(xc=self.xc, yc=self.yc, radius=self.radius)

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(xc=rec['xc'], yc=rec['yc'], radius=rec['radius'])


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

        :param thresh: An optional threshhold. If present, the vertex
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
        return dict(vx=np.asarray(self.vx).tolist(),
                    vy=np.asarray(self.vy).tolist())

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(vx=rec['vx'], vy=rec['vy'])


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
        if isinstance(x[0], np.datetime64)\
                or 'datetime64' in str(type(x[0]))\
                or isinstance(x[0], datetime.date):
            x = pd.to_datetime(x)
            x = mdate.date2num(x)
            vx = pd.to_datetime(self.vx)
            vx = mdate.date2num(vx)
        else:
            vx = self.vx
        xypts = np.column_stack((x.flat, y.flat))
        xyvts = np.column_stack((vx, self.vy))
        result = points_inside_poly(xypts, xyvts)
        good = np.isfinite(xypts).all(axis=1)
        result[~good] = False
        result.shape = x.shape
        return result


class Path(VertexROIBase):

    def __str__(self):
        result = 'Path ('
        result += ','.join(['(%s, %s)' % (x, y)
                            for x, y in zip(self.vx, self.vy)])
        result += ')'
        return result


class AbstractMplRoi(object):  # pragma: no cover

    """ Base class for objects which use
    Matplotlib user events to edit/display ROIs
    """

    def __init__(self, axes):
        """
        :param axes: The Matplotlib Axes object to draw to
        """

        self._axes = axes
        self._roi = self._roi_factory()
        self._mid_selection = False

    def _draw(self):
        self._axes.figure.canvas.draw()

    def _roi_factory(self):
        raise NotImplementedError()

    def roi(self):
        return self._roi.copy()

    def reset(self):
        self._roi.reset()
        self._mid_selection = False
        self._sync_patch()

    def active(self):
        return self._mid_selection

    def start_selection(self, event):
        raise NotImplementedError()

    def update_selection(self, event):
        raise NotImplementedError()

    def finalize_selection(self, event):
        raise NotImplementedError()

    def _sync_patch(self):
        raise NotImplementedError()


class MplPickROI(AbstractMplRoi):

    def _draw(self):
        pass

    def _roi_factory(self):
        return PointROI()

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
    A subclass of RectangularROI that also renders the ROI to a plot

    *Attributes*:

        plot_opts:

                   Dictionary instance
                   A dictionary of plot keywords that are passed to
                   the patch representing the ROI. These control
                   the visual properties of the ROI
    """

    def __init__(self, axes):
        """
        :param axes: A matplotlib Axes object to attach the graphical ROI to
        """

        AbstractMplRoi.__init__(self, axes)

        self._xi = None
        self._yi = None
        self._scrubbing = False

        self.plot_opts = {'edgecolor': PATCH_COLOR, 'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._patch = Rectangle((0., 0.), 1., 1.)
        self._patch.set_zorder(100)
        self._setup_patch()

    def _setup_patch(self):
        self._axes.add_patch(self._patch)
        self._patch.set_visible(False)

        self._sync_patch()

    def _roi_factory(self):
        return RectangularROI()

    def start_selection(self, event):
        if event.inaxes != self._axes:
            return

        self._xi = event.xdata
        self._yi = event.ydata

        if self._roi.defined() and \
                self._roi.contains(event.xdata, event.ydata):
            self._scrubbing = True
            self._cx, self._cy = self._roi.center()
        else:
            self._scrubbing = False
            self._roi.reset()
            self._roi.update_limits(event.xdata, event.ydata,
                                    event.xdata, event.ydata)

        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection or event.inaxes != self._axes:
            return

        if self._scrubbing:
            self._roi.move_to(self._cx + event.xdata - self._xi,
                              self._cy + event.ydata - self._yi)
        else:
            self._roi.update_limits(min(event.xdata, self._xi),
                                    min(event.ydata, self._yi),
                                    max(event.xdata, self._xi),
                                    max(event.ydata, self._yi))
        self._sync_patch()

    def finalize_selection(self, event):
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
        self._draw()

    def __str__(self):
        return "MPL Rectangle: %s" % self._patch


class MplXRangeROI(AbstractMplRoi):

    def __init__(self, axes):
        """
        :param axes: A matplotlib Axes object to attach the graphical ROI to
        """

        AbstractMplRoi.__init__(self, axes)
        self._xi = None

        self.plot_opts = {'edgecolor': PATCH_COLOR, 'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        trans = blended_transform_factory(self._axes.transData,
                                          self._axes.transAxes)
        self._patch = Rectangle((0., 0.), 1., 1., transform=trans)
        self._patch.set_zorder(100)
        self._setup_patch()

    def _setup_patch(self):
        self._axes.add_patch(self._patch)
        self._patch.set_visible(False)
        self._sync_patch()

    def _roi_factory(self):
        return XRangeROI()

    def start_selection(self, event):
        if event.inaxes != self._axes:
            return

        self._roi.reset()
        self._roi.set_range(event.xdata, event.xdata)
        self._xi = event.xdata
        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection or event.inaxes != self._axes:
            return
        self._roi.set_range(min(event.xdata, self._xi),
                            max(event.xdata, self._xi))
        self._sync_patch()

    def finalize_selection(self, event):
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
        self._draw()


class MplYRangeROI(AbstractMplRoi):

    def __init__(self, axes):
        """
        :param axes: A matplotlib Axes object to attach the graphical ROI to
        """

        AbstractMplRoi.__init__(self, axes)
        self._xi = None

        self.plot_opts = {'edgecolor': PATCH_COLOR, 'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        trans = blended_transform_factory(self._axes.transAxes,
                                          self._axes.transData)
        self._patch = Rectangle((0., 0.), 1., 1., transform=trans)
        self._patch.set_zorder(100)
        self._setup_patch()

    def _setup_patch(self):
        self._axes.add_patch(self._patch)
        self._patch.set_visible(False)
        self._sync_patch()

    def _roi_factory(self):
        return YRangeROI()

    def start_selection(self, event):
        if event.inaxes != self._axes:
            return

        self._roi.reset()
        self._roi.set_range(event.ydata, event.ydata)
        self._xi = event.ydata
        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection or event.inaxes != self._axes:
            return
        self._roi.set_range(min(event.ydata, self._xi),
                            max(event.ydata, self._xi))
        self._sync_patch()

    def finalize_selection(self, event):
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
        self._draw()


class MplCircularROI(AbstractMplRoi):

    """
    Class to display / edit circular ROIs using matplotlib

    Since circles on the screen may not be circles in the data
    (due, e.g., to logarithmic scalings on the axes), the
    ultimate ROI that is created is a polygonal ROI

    :param plot_opts:

               A dictionary of plot keywords that are passed to
               the patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, axes):
        """
        :param axes: A matplotlib Axes object to attach the graphical ROI to
        """

        AbstractMplRoi.__init__(self, axes)
        self.plot_opts = {'edgecolor': PATCH_COLOR, 'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._xi = None
        self._yi = None
        self._setup_patch()

    def _setup_patch(self):
        self._patch = Ellipse((0., 0.), transform=IdentityTransform(),
                              width=0., height=0.,)
        self._patch.set_zorder(100)
        self._patch.set(**self.plot_opts)
        self._axes.add_patch(self._patch)
        self._patch.set_visible(False)
        self._sync_patch()

    def _roi_factory(self):
        return CircularROI()

    def _sync_patch(self):
        # Update geometry
        if not self._roi.defined():
            self._patch.set_visible(False)
        else:
            xy = self._roi.get_center()
            r = self._roi.get_radius()
            self._patch.center = xy
            self._patch.width = 2. * r
            self._patch.height = 2. * r
            self._patch.set_visible(True)

        # Update appearance
        self._patch.set(**self.plot_opts)

        # Refresh
        self._axes.figure.canvas.draw()

    def start_selection(self, event):
        if event.inaxes != self._axes:
            return

        xy = data_to_pixel(self._axes, [event.xdata], [event.ydata])
        self._roi.set_center(xy[0, 0], xy[0, 1])
        self._roi.set_radius(0.)
        self._xi = xy[0, 0]
        self._yi = xy[0, 1]
        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection or event.inaxes != self._axes:
            return

        xy = data_to_pixel(self._axes, [event.xdata], [event.ydata])
        dx = xy[0, 0] - self._xi
        dy = xy[0, 1] - self._yi
        self._roi.set_radius(np.hypot(dx, dy))
        self._sync_patch()

    def roi(self):
        if not self._roi.defined():
            return PolygonalROI()

        theta = np.linspace(0, 2 * np.pi, num=200)
        xy_center = self._roi.get_center()
        rad = self._roi.get_radius()
        x = xy_center[0] + rad * np.cos(theta)
        y = xy_center[1] + rad * np.sin(theta)
        xy_data = pixel_to_data(self._axes, x, y)
        vx = xy_data[:, 0].ravel().tolist()
        vy = xy_data[:, 1].ravel().tolist()
        result = PolygonalROI(vx, vy)
        return result

    def finalize_selection(self, event):
        self._mid_selection = False
        self._patch.set_visible(False)
        self._axes.figure.canvas.draw()


class MplPolygonalROI(AbstractMplRoi):

    """
    Defines and displays polygonal ROIs on matplotlib plots

    Attributes:

        plot_opts: Dictionary instance
                   A dictionary of plot keywords that are passed to
                   the patch representing the ROI. These control
                   the visual properties of the ROI
    """

    def __init__(self, axes):
        """
        :param axes: A matplotlib Axes object to attach the graphical ROI to
        """
        AbstractMplRoi.__init__(self, axes)
        self.plot_opts = {'edgecolor': PATCH_COLOR, 'facecolor': PATCH_COLOR,
                          'alpha': 0.3}

        self._setup_patch()

    def _setup_patch(self):
        self._patch = Polygon(np.array(list(zip([0, 1], [0, 1]))))
        self._patch.set_zorder(100)
        self._patch.set(**self.plot_opts)
        self._axes.add_patch(self._patch)
        self._patch.set_visible(False)
        self._sync_patch()

    def _roi_factory(self):
        return PolygonalROI()

    def _sync_patch(self):
        # Update geometry
        if not self._roi.defined():
            self._patch.set_visible(False)
        else:
            x, y = self._roi.to_polygon()
            self._patch.set_xy(list(zip(x + [x[0]],
                                        y + [y[0]])))
            self._patch.set_visible(True)

        # Update appearance
        self._patch.set(**self.plot_opts)

        # Refresh
        self._axes.figure.canvas.draw()

    def start_selection(self, event):
        if event.inaxes != self._axes:
            return

        self._roi.reset()
        self._roi.add_point(event.xdata, event.ydata)
        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection or event.inaxes != self._axes:
            return
        self._roi.add_point(event.xdata, event.ydata)
        self._sync_patch()

    def finalize_selection(self, event):
        self._mid_selection = False
        self._patch.set_visible(False)
        self._axes.figure.canvas.draw()


class MplPathROI(MplPolygonalROI):

    def roi_factory(self):
        return Path()

    def _setup_patch(self):
        self._patch = None

    def _sync_patch(self):
        if self._patch is not None:
            self._patch.remove()
            self._patch = None

        # Update geometry
        if not self._roi.defined():
            return
        else:
            x, y = self._roi.to_polygon()
            p = MplPath(np.column_stack((x, y)))
            self._patch = PatchPath(p)
            self._patch.set_visible(True)

        # Update appearance
        self._patch.set(**self.plot_opts)

        # Refresh
        self._axes.figure.canvas.draw()

    def finalize_selection(self, event):
        self._mid_selection = False
        if self._patch is not None:
            self._patch.set_visible(False)
        self._axes.figure.canvas.draw()
