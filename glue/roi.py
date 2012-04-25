import numpy as np
from matplotlib.nxutils import points_inside_poly

np.seterr(all='ignore')

from matplotlib.patches import Polygon, Rectangle, Ellipse
from glue.exceptions import UndefinedROI

def aspect_ratio(ax):
    """ Returns the pixel height / width of a box that spans 1
    data unit in x and y
    """
    width = ax.get_position().width * ax.figure.get_figwidth()
    height = ax.get_position().height * ax.figure.get_figheight()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return height / width / (ymax - ymin) * (xmax - xmin)


def data_to_norm(ax, x, y):
    xy = np.column_stack( (np.asarray(x).flat, np.asarray(y).flat))
    pixel = ax.transData.transform(xy)
    norm = ax.transAxes.inverted().transform(pixel)
    return norm

def data_to_pixel(ax, x, y):
    xy = np.column_stack( (np.asarray(x).flat, np.asarray(y).flat))
    return ax.transData.transform(xy)

def pixel_to_data(ax, x, y):
    xy = np.column_stack( (np.asarray(x).flat, np.asarray(y).flat))
    return ax.transData.inverted().transform(xy)


class Roi(object):
    def contains(self, x, y):
        """Return true/false for each x/y pair. Raises UndefinedROI
        exception if not defined
        """
        raise NotImplementedError()

    def defined(self):
        """ Returns whether or not the subset is properly defined """
        raise NotImplementedError()

    def to_polygon(self):
        """ Returns a tuple of x and y points, approximating the ROI
        as a polygon."""
        raise NotImplementedError

class RectangularROI(Roi):
    """
    A class to define a 2D rectangular region of interest.
    """

    def __init__(self):
        """ Create a new ROI """

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def __str__(self):
        return "x=[%0.3f, %0.3f], y=[%0.3f, %0.3f]" % (self.xmin, self.xmax,
                                                       self.ymin, self.ymax)
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

        Parameters:
        -----------
        x: A scalar or numpy array of x points
        y: A scalar or numpy array of y points

        Returns:
        --------
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
        return [self.xmin, self.xmax, self.xmax, self.xmin, self.xmin], \
            [self.ymin, self.ymin, self.ymax, self.ymax, self.ymin]

class CircularROI(Roi):
    """
    A class to define a 2D circular region of interest.
    """

    def __init__(self):
        """ Create a new ROI """
        self.xc = None
        self.yc = None
        self.radius = None

    def contains(self, x, y):
        """
        Test whether a set of (x,y) points falls within
        the region of interest

        Parameters:
        -----------
        x: A list of x points
        y: A list of y points

        Returns:
        --------
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
        return self.xc is not None and \
            self.yc is not None and self.radius is not None

    def to_polygon(self):
        theta = np.linspace(0, 2 * np.pi, num = 20)
        x = self.xc + self.radius * np.cos(theta)
        y = self.yc + self.radius * np.sin(theta)
        return x, y

class PolygonalROI(Roi):
    """
    A class to define 2D polygonal regions-of-interest
    """

    def __init__(self, vx=None, vy=None):
        """
        Create a new ROI
        """
        self.vx = vx
        self.vy = vy
        if self.vx is None:
            self.vx = []
        if self.vy is None:
            self.vy = []

    def __str__(self):
        result = 'Polygonal ROI ('
        result += ','.join(['(%s, %s)' % (x,y)
                            for x,y in zip(self.vx, self.vy)])
        result += ')'
        return result

    def contains(self, x, y):
        """
        Test whether a set of (x,y) points falls within
        the region of interest

        Parameters:
        -----------
        x: A list of x points
        y: A list of y points

        Returns:
        --------
        A list of True/False values, for whether each (x,y)
        point falls within the ROI
        """
        if not self.defined():
            raise UndefinedROI
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        xypts = np.column_stack((x.flat,y.flat))
        xyvts = np.column_stack((self.vx, self.vy))
        result = points_inside_poly(xypts, xyvts)
        result.shape = x.shape
        return result

    def add_point(self, x, y):
        """
        Add another vertex to the ROI

        Parameters:
        -----------
        x: The x coordinate
        y: The y coordinate
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
        """
        Remove the vertex closest to a reference (xy) point

        Parameters
        ----------
        x: The x coordinate of the reference point
        y: The y coordinate of the reference point
        thresh: An optional threshhold. If present, the vertex closest
                to (x,y) will only be removed if the distance is less
                than thresh

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

class AbstractMplRoi(object):
    """ Base class for objects which use
    Matplotlib user events to edit/display ROIs
    """
    def __init__(self, ax):

        self._ax = ax
        self._roi = self._roi_factory()

    def _roi_factory(self):
        raise NotImplementedError()

    def roi(self):
        return self._roi

    def start_selection(self, event):
        raise NotImplementedError()

    def _update_seleplction(self, event):
        raise NotImplementedError()

    def finalize_selection(self, event):
        raise NotImplementedError()

    def _reset(self):
        raise NotImplementedError()

    def _sync_patch(self):
        raise NotImplementedError()

class MplRectangularROI(AbstractMplRoi):
    """
    A subclass of RectangularROI that also renders the ROI to a plot

    Attributes:
    -----------
    plot_opts: Dictionary instance
               A dictionary of plot keywords that are passed to
               the patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, ax):
        """
        Create a new ROI

        Parameters
        ----------
        ax: A matplotlib Axes object to attach the graphical ROI to
        """

        AbstractMplRoi.__init__(self, ax)

        self._mid_selection = False
        self._xi = None
        self._yi = None

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._rectangle = Rectangle((0., 0.), 1., 1.)
        self._ax.add_patch(self._rectangle)

        self._sync_patch()

    def _roi_factory(self):
        return RectangularROI()


    def start_selection(self, event):
        if not (event.inaxes):
            return

        self._roi.reset()
        self._roi.update_limits(event.xdata, event.ydata,
                                event.xdata, event.ydata)
        self._xi = event.xdata
        self._yi = event.ydata

        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection:
            return

        self._roi.update_limits(min(event.xdata, self._xi),
                                min(event.ydata, self._yi),
                                max(event.xdata, self._xi),
                                max(event.ydata, self._yi))
        self._sync_patch()

    def finalize_selection(self, event):
        self._mid_selection = False
        self._rectangle.set_visible(False)
        self._ax.figure.canvas.draw()

    def _sync_patch(self):
        if self._roi.defined():
            corner = self._roi.corner()
            width = self._roi.width()
            height = self._roi.height()
            self._rectangle.set_xy(corner)
            self._rectangle.set_width(width)
            self._rectangle.set_height(height)
            self._rectangle.set(**self.plot_opts)
            self._rectangle.set_visible(True)
        else:
            self._rectangle.set_visible(False)
        self._ax.figure.canvas.draw()

    def __str__(self):
        return "MPL Rectangle: %s" % self._rectangle

class MplCircularROI(AbstractMplRoi):
    """
    Class to display / edit circular ROIs using matplotlib

    Since circles on the screen may not be circles in the data
    (due, e.g., to logarithmic scalings on the axes), the
    ultimate ROI that is created is a polygonal ROI

    Attributes:
    -----------
    plot_opts: Dictionary instance
               A dictionary of plot keywords that are passed to
               the patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, ax):
        """
        Create a new ROI

        Parameters
        ----------
        ax: A matplotlib Axes object to attach the graphical ROI to
        """

        AbstractMplRoi.__init__(self, ax)
        self._mid_selection = False
        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._xi = None
        self._yi = None

        self._circle = Ellipse((0., 0.), transform=None,
                               width=0., height=0.,)
        self._circle.set(**self.plot_opts)
        self._ax.add_patch(self._circle)

        self._sync_patch()

    def _roi_factory(self):
        return CircularROI()

    def _sync_patch(self):
        # Update geometry
        if not self._roi.defined():
            self._circle.set_visible(False)
        else:
            xy = self._roi.get_center()
            r = self._roi.get_radius()
            self._circle.center = xy
            self._circle.width = 2. * r
            self._circle.height = 2. * r
            self._circle.set_visible(True)

        # Update appearance
        self._circle.set(**self.plot_opts)

        # Refresh
        self._ax.figure.canvas.draw()

    def start_selection(self, event):
        if not (event.inaxes):
            return

        xy = data_to_pixel(self._ax, [event.xdata], [event.ydata])
        self._roi.set_center(xy[0,0], xy[0,1])
        self._roi.set_radius(0.)
        self._xi = xy[0,0]
        self._yi = xy[0,1]

        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):

        if not self._mid_selection:
            return

        xy = data_to_pixel(self._ax, [event.xdata], [event.ydata])
        dx = xy[0,0] - self._xi
        dy = xy[0,1] - self._yi
        self._roi.set_radius(np.sqrt(dx * dx + dy * dy))
        self._sync_patch()

    def roi(self):
        theta = np.linspace(0, 2 * np.pi, num = 200)
        xy_center = self._roi.get_center()
        rad = self._roi.get_radius()
        x = xy_center[0] + rad * np.cos(theta)
        y = xy_center[1] + rad * np.sin(theta)
        xy = np.zeros( (x.size, 2))
        xy[:, 0] = x
        xy[:, 1] = y
        xy_data = self._ax.transData.inverted().transform(xy)
        vx = xy_data[:,0].flatten().tolist()
        vy = xy_data[:,1].flatten().tolist()
        result = PolygonalROI(vx, vy)
        return result

    def finalize_selection(self, event):
        self._mid_selection = False
        self._circle.set_visible(False)
        self._ax.figure.canvas.draw()

class MplPolygonalROI(AbstractMplRoi):
    """
    Defines and displays polygonal ROIs on matplotlib plots

    Attributes:
    -----------
    plot_opts: Dictionary instance
               A dictionary of plot keywords that are passed to
               the patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, ax):
        """
        Parameters
        ----------
        ax: A matplotlib Axes object to attach the graphical ROI to
        """
        AbstractMplRoi.__init__(self, ax)
        self._mid_selection = False
        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'red',
                          'alpha': 0.3}
        self._polygon = Polygon(np.array(zip([0, 1], [0, 1])))
        self._polygon.set(**self.plot_opts)

        self._ax.add_patch(self._polygon)
        self._sync_patch()

    def _roi_factory(self):
        return PolygonalROI()

    def _sync_patch(self):
        # Update geometry
        if not self._roi.defined():
            self._polygon.set_visible(False)
        else:
            x,y = self._roi.to_polygon()
            self._polygon.set_xy(zip(x + [x[0]],
                                     y + [y[0]]))
            self._polygon.set_visible(True)

        # Update appearance
        self._polygon.set(**self.plot_opts)

        # Refresh
        self._ax.figure.canvas.draw()

    def start_selection(self, event):
        if not (event.inaxes):
            return

        self._roi.reset()
        self._roi.add_point(event.xdata, event.ydata)
        self._mid_selection = True
        self._sync_patch()

    def update_selection(self, event):
        if not self._mid_selection or not event.inaxes:
            return
        self._roi.add_point(event.xdata, event.ydata)
        self._sync_patch()

    def finalize_selection(self, event):
        self._mid_selection = False
        self._polygon.set_visible(False)
        self._ax.figure.canvas.draw()

