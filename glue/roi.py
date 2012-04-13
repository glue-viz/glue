import numpy as np
np.seterr(all='ignore')

from matplotlib.patches import Polygon, Rectangle, Ellipse


def aspect_ratio(ax):
    width = ax.get_position().width * ax.figure.get_figwidth()
    height = ax.get_position().height * ax.figure.get_figheight()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return height / width / (ymax - ymin) * (xmax - xmin)


def data_to_norm(ax, x, y):
    pixel = ax.transData(zip(x, y))
    norm = ax.transAxes.inverted()(pixel)
    return zip(*norm)

class Roi(object):
    def contains(self, x, y):
        raise NotImplementedError()

    def defined(self):
        raise NotImplementedError()

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


class CircularROI(Roi):
    """
    A class to define a 2D circular region of interest.
    """

    def __init__(self):
        """ Create a new ROI """
        self.xc = None
        self.yc = None
        self.radius = 0.

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

    def reset(self):
        """
        Reset the rectangular region.
        """
        self.xc = None
        self.yc = None
        self.radius = 0.

    def defined(self):
        return self.xc is not None


class PolygonalROI(Roi):
    """
    A class to define 2D polygonal regions-of-interest
    """

    def __init__(self):
        """
        Create a new ROI
        """
        self.vx = []
        self.vy = []

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
        result = np.zeros(x.shape, dtype=int)

        # Treat special case of empty ROI
        if len(self.vx) == 0:
            return result

        xi = np.array(self.vx)
        xj = np.roll(xi, 1)
        yi = np.array(self.vy)
        yj = np.roll(yi, 1)

        from time import time
        t0 = time()
        for i in range(len(xi)):
            result += ((yi[i] > y) != (yj[i] > y)) & \
                       (x < (xj[i] - xi[i]) * (y - yi[i])
                        / (yj[i] - yi[i]) + xi[i])
        t1 = time()
        print t1 - t0
        return np.mod(result, 2) == 1

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

class MplCircularROI(CircularROI, AbstractMplRoi):
    """
    A subclass of CircularROI that also renders the ROI to a plot

    Attributes:
    -----------
    plot_opts: Dictionary instance
               A dictionary of plot keywords that are passed to
               the patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, ax, subset=None):
        """
        Create a new ROI

        Parameters
        ----------
        ax: A matplotlib Axes object to attach the graphical ROI to
        """

        CircularROI.__init__(self)
        AbstractMplRoi.__init__(self, ax, subset=subset)

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._circle = Ellipse((0., 0.), width=0., height=0.)

        self._circle.set(**self.plot_opts)

        self._ax.add_patch(self._circle)

        self.connect()
        self._sync_patch()

    def _sync_patch(self):

        # Update geometry
        if not self.defined():
            self._circle.set_visible(False)
        else:
            self._circle.center = (self.xc, self.yc)
            self._circle.width = 2. * self.radius
            self._circle.height = 2. * self.radius / aspect_ratio(self._ax)
            self._circle.set_visible(True)

        # Update appearance
        self._circle.set(**self.plot_opts)

        # Refresh
        self._ax.figure.canvas.draw()

    def set_center(self, x, y):
        CircularROI.set_center(self, x, y)
        self._sync_patch()

    def set_radius(self, radius):
        CircularROI.set_radius(self, radius)
        self._sync_patch()

    def start_selection(self, event):

        if not (event.inaxes):
            return

        self._reset()
        self.set_center(event.xdata, event.ydata)
        self.set_radius(0.)

        self._xi = event.xdata
        self._yi = event.ydata

        self._mid_selection = True

    def update_selection(self, event):

        if not (event.inaxes and self._mid_selection):
            return

        dx = event.xdata - self._xi
        dy = (event.ydata - self._yi) * aspect_ratio(self._ax)

        self.set_radius(np.sqrt(dx * dx + dy * dy))

    def to_polygon(self):
        ar = aspect_ratio(self._ax)
        self._circle.center = (self.xc, self.yc)
        self._circle.width = 2. * self.radius
        self._circle.height = 2. * self.radius / ar
        theta = np.linspace(0, 2 * np.pi, num = 20)
        x = self.xc + self._circle.width * np.cos(theta) / 2.
        y = self.yc + self._circle.height * np.sin(theta) / 2.
        result = PolygonalROI()
        result.vx = x
        result.vy = y
        return result


class MplPolygonalROI(PolygonalROI, AbstractMplRoi):
    """
    A subclass of PolygonalROI that also renders the ROI to a plot

    Attributes:
    -----------
    plot_opts: Dictionary instance
               A dictionary of plot keywords that are passed to
               the patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, ax, lasso = True, subset=None):
        """
        Create a new ROI

        Parameters
        ----------
        ax: A matplotlib Axes object to attach the graphical ROI to
        """

        PolygonalROI.__init__(self)
        AbstractMplRoi.__init__(self, ax, subset=subset)

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'red',
                          'alpha': 0.3}

        self._polygon = Polygon(np.array(zip([0, 1], [0, 1])))
        self._polygon.set(**self.plot_opts)
        self.vx = [0.]
        self.vy = [0.]

        self._ax.add_patch(self._polygon)
        self.lasso = lasso
        self.connect()
        self._sync_patch()

    def connect(self):
        if self.lasso:
            AbstractMplRoi.connect(self)
            return

        self.disconnect()
        canvas = self._ax.figure.canvas
        self._press = canvas.mpl_connect('button_press_event',
                                         self.update_selection)
        self._release = canvas.mpl_connect('button_release_event',
                                           self.finalize_selection)

    def _sync_patch(self):
        # Update geometry
        if not self.defined():
            self._polygon.set_visible(False)
        else:
            self._polygon.set_xy(np.array(zip(self.vx + [self.vx[0]],
                                              self.vy + [self.vy[0]])))
            self._polygon.set_visible(True)

        # Update appearance
        self._polygon.set(**self.plot_opts)

        # Refresh
        self._ax.figure.canvas.draw()

    def add_point(self, x, y):
        PolygonalROI.add_point(self, x, y)
        self._sync_patch()

    def _reset(self):
        PolygonalROI._reset(self)
        self._sync_patch()

    def replace_last_point(self, x, y):
        PolygonalROI.replace_last_point(self, x, y)
        self._sync_patch()

    def remove_point(self, x, y, thresh=None):
        PolygonalROI.remove_point(self, x, y, thresh=None)
        self._sync_patch()

    def start_selection(self, event):
        if not (event.inaxes):
            return
        self._mid_selection = True
        self._reset()
        self.add_point(event.xdata, event.ydata)

    def update_selection(self, event):
        if not self._mid_selection:
            return
        self.add_point(event.xdata, event.ydata)

    def to_polygon(self):
        result = PolygonalROI()
        result.vx = self.vx
        result.vy = self.vy
        return result
