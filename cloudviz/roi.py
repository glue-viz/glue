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

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlog = ax.get_xscale() == 'log'
    ylog = ax.get_yscale() == 'log'

    if not xlog:
        xnorm = (x - xlim[0]) / (xlim[1] - xlim[0])
    else:
        xnorm = np.log(x / xlim[0]) / np.log(xlim[1] / xlim[0])

    if not ylog:
        ynorm = (y - ylim[0]) / (ylim[1] - ylim[0])
    else:
        ynorm = np.log(y / ylim[0]) / np.log(ylim[1] / ylim[0])
    
    return xnorm, ynorm

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
        return (x > self.xmin) & (x < self.xmax) & \
               (y > self.ymin) & (y < self.ymax)

    def update_limits(self, xmin, ymin, xmax, ymax):
        """
        Update the limits of the rectangle
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

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

        for i in range(len(xi)):
            result += ((yi[i] > y) != (yj[i] > y)) & \
                       (x < (xj[i] - xi[i]) * (y - yi[i])
                        / (yj[i] - yi[i]) + xi[i])

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
    """ Base class for objects which display
    ROIs on Matplotlib plots and, optionally,
    edit subset ROIs 
    """
    def __init__(self, ax, subset=None):

        self._ax = ax
        self._mid_selection = False
        self._subset = subset

        self._press = None  # id of MPL connection to button_press
        self._motion = None # id of MPL connection to motion_notify
        self._release = None # id of MPL connection to button_release
        
    def start_selection(self, event):
        raise NotImplementedError()

    def update_selection(self, event):
        raise NotImplementedError()

    def finalize_selection(self, event):
        if not (self._mid_selection):
            return
        if self._subset is not None:
            self._subset.roi = self.to_polygon()
        self.reset()
        self._mid_selection = False
        self._sync_patch()
            
    def connect(self):
        self.disconnect()
        canvas = self._ax.figure.canvas
        self._press = canvas.mpl_connect('button_press_event',
                                         self.start_selection)
        self._motion = canvas.mpl_connect('motion_notify_event',
                                          self.update_selection)
        self._release = canvas.mpl_connect('button_release_event',
                                           self.finalize_selection)

    def disconnect(self):
        if self._press is not None:
            self._ax.figure.canvas.mpl_disconnect(self._press)
            self._press = None
        if self._motion is not None:
            self._ax.figure.canvas.mpl_disconnect(self._motion)
            self._motion = None
        if self._release is not None:
            self._ax.figure.canvas.mpl_disconnect(self._release)
            self._release = None

    def to_polygon(self):
        raise NotImplementedError()
        

class MplRectangularROI(RectangularROI, AbstractMplRoi):
    """
    A subclass of RectangularROI that also renders the ROI to a plot

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

        RectangularROI.__init__(self)
        AbstractMplRoi.__init__(self, ax, subset=subset)

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._rectangle = Rectangle((0., 0.), 1., 1.)
        self._rectangle.set(**self.plot_opts)

        self._ax.add_patch(self._rectangle)

        self._sync_patch()
        self.connect()

    def _sync_patch(self):

        # Update geometry
        if self.xmin is None:
            self._rectangle.set_visible(False)
        else:
            self._rectangle.set_xy((self.xmin, self.ymin))
            self._rectangle.set_width(self.xmax - self.xmin)
            self._rectangle.set_height(self.ymax - self.ymin)
            self._rectangle.set_visible(True)

        # Update appearance
        self._rectangle.set(**self.plot_opts)

        # Refresh
        self._ax.figure.canvas.draw()

    def update_limits(self, xmin, ymin, xmax, ymax):
        RectangularROI.update_limits(self, xmin, ymin, xmax, ymax)
        self._sync_patch()

    def start_selection(self, event):        
        if not (event.inaxes):
            return

        self.reset()
        self.update_limits(event.xdata, event.ydata, event.xdata, event.ydata)

        self._xi = event.xdata
        self._yi = event.ydata

        self._mid_selection = True

    def update_selection(self, event):

        if not self._mid_selection:
            return

        self.update_limits(min(event.xdata, self._xi),
                           min(event.ydata, self._yi),
                           max(event.xdata, self._xi),
                           max(event.ydata, self._yi))
        
    def to_polygon(self):
        x = [self.xmin, self.xmax, self.xmax, self.xmin]
        y = [self.ymin, self.ymin, self.ymax, self.ymax]
        result = PolygonalROI()
        result.vx = x
        result.vy = y
        return result

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

        self.reset()
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
        theta = np.linspace(0, 2 * np.pi, num = 100)
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

    def reset(self):
        PolygonalROI.reset(self)
        self._sync_patch()

    def replace_last_point(self, x, y):
        PolygonalROI.replace_last_point(self, x, y)
        self._sync_patch()

    def remove_point(self, x, y, thresh=None):
        PolygonalROI.reset(self, x, y, thresh=None)
        self._sync_patch()

    def start_selection(self, event):
        if not (event.inaxes):
            return
        self._mid_selection = True
        self.reset()
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


class MplTreeROI(AbstractMplRoi):
    def __init__(self, ax, single=False, subset=None):

        AbstractMplRoi.__init__(self, ax, subset=subset)
        self.single = single
        self.connect()

    def start_selection(self, event):
        if not (event.inaxes):
            return

        self._mid_selection = True

    def update_selection(self, event):
        if not (event.inaxes and self._mid_selection):
            return

        subset = self.get_subset()
        if subset is None: return

        x = event.xdata
        y = event.ydata
        
        #find the closest point to xy
        xx = self._data.components[self._component_x].data
        yy = self._data.components[self._component_y].data
        
        xxn, yyn = data_to_norm(self._ax, xx, yy)
        xn, yn = data_to_norm(self._ax, x, y)

        xbad = np.isnan(xxn)
        ybad = np.isnan(yyn)
        xxn[xbad] = 10 * np.nanmax(xxn)
        yyn[ybad] = 10 * np.nanmax(yyn)

        d = [(xn - xxn[i])**2 + (yn - yyn[i])**2 for i in range(len(xxn))]
        best = np.argmin(d)
        index = self._data.tree.index_map.flatten()[best]

        if self.single:
            id = [index]
        else:
            id = subset.data.tree._index[index].get_subtree_indices()

        subset.node_list = id
    
    def finalize_selection(self, event):
        if not (event.inaxes and self._mid_selection):
            return
        self.update_selection(event)
        
        self._mid_selection = False
