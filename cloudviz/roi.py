import numpy as np
np.seterr(all='ignore')
from matplotlib.patches import Polygon, Rectangle, Ellipse

import cloudviz as cv

class RectangularROI(object):
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


class CircularROI(object):
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


class PolygonalROI(object):
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
        if not self.vx:
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
        if not self.vx:
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

class AbstractMplRoiTool(object):
    def __init__(self, data, component_x, component_y, ax):

        self._ax = ax
        self._mid_selection = False
        self._active = True
        self._data = data

        self.set_x_attribute(component_x)
        self.set_y_attribute(component_y)
        
        self._press = None  # id of MPL connection to button_press
        self._motion = None # id of MPL connection to motion_notify
        self._release = None # id of MPL connection to button_release
        self.connect()
        
    def get_subset(self):
        subset = self._data.get_active_subset()
        if subset is None: return subset

        if not self.check_compatible_subset(subset):
            raise TypeError("ROI and Subset are incompatible: %s, %s" %
                            (type(self), type(subset)))
        return subset

    def set_x_attribute(self, attribute):
        if attribute not in self._data.components:
            raise KeyError("%s is not a valid component" % attribute)
        self._component_x = attribute

    def set_y_attribute(self, attribute):
        if attribute not in self._data.components:
            raise KeyError("%s is not a valid component" % attribute)
        self._component_y = attribute


    def check_compatible_subset(self, subset):
        raise NotImplementedError()

    def set_active(self, state):
        self._active = state

    def start_selection(self, event):
        raise NotImplementedError()

    def update_selection(self, event):
        raise NotImplementedError()

    def finalize_selection(self, event):
        raise NotImplementedError()
        
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
        

class MplRectangularROI(RectangularROI):
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

        RectangularROI.__init__(self)

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._rectangle = Rectangle((0., 0.), 1., 1.)
        self._rectangle.set(**self.plot_opts)

        self._ax = ax
        self._ax.add_patch(self._rectangle)

        self._sync_patch()

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

    def reset(self):
        RectangularROI.reset(self)
        self._sync_patch()


def aspect_ratio(ax):
    width = ax.get_position().width * ax.figure.get_figwidth()
    height = ax.get_position().height * ax.figure.get_figheight()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return height / width / (ymax - ymin) * (xmax - xmin)


class MplCircularROI(CircularROI):
    """
    A subclass of CircularROI that also renders the ROI to a plot

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

        CircularROI.__init__(self)

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._circle = Ellipse((0., 0.), width=0., height=0.)
        self._circle.set(**self.plot_opts)

        self._ax = ax
        self._ax.add_patch(self._circle)

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

    def reset(self):
        CircularROI.reset(self)
        self._sync_patch()


class MplPolygonalROI(PolygonalROI):
    """
    A subclass of PolygonalROI that also renders the ROI to a plot

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

        PolygonalROI.__init__(self)

        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}

        self._polygon = Polygon(np.array(zip([0, 1], [0, 1])))
        self._polygon.set(**self.plot_opts)

        self._ax = ax
        self._ax.add_patch(self._polygon)

        self._sync_patch()

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


class MplBoxTool(MplRectangularROI, AbstractMplRoiTool):

    def __init__(self, data, component_x, component_y, ax):

        MplRectangularROI.__init__(self, ax)
        AbstractMplRoiTool.__init__(self, data, component_x, component_y, ax)

    def check_compatible_subset(self, subset):
        return isinstance(subset, cv.subset.ElementSubset)
        
    def start_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._active:
            return

        self.reset()
        self.update_limits(event.xdata, event.ydata, event.xdata, event.ydata)

        self._xi = event.xdata
        self._yi = event.ydata

        self._mid_selection = True

    def update_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._active:
            return

        if not self._mid_selection:
            return

        self.update_limits(min(event.xdata, self._xi),
                           min(event.ydata, self._yi),
                           max(event.xdata, self._xi),
                           max(event.ydata, self._yi))

    def finalize_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._mid_selection:
            return

        subset = self.get_subset()
        if subset is None:
            return
        x = self._data.components[self._component_x].data
        y = self._data.components[self._component_y].data

        subset.mask = self.contains(x, y)

        self.reset()

        self._mid_selection = False


class MplCircleTool(MplCircularROI, AbstractMplRoiTool):

    def __init__(self, data, component_x, component_y, ax):

        MplCircularROI.__init__(self, ax)
        AbstractMplRoiTool.__init__(self, data, component_x, component_y, ax)
        
    def check_compatible_subset(self, subset):
        return isinstance(subset, cv.subset.ElementSubset)

    def start_selection(self, event):

        if not (event.inaxes and self._active):
            return

        self.reset()
        self.set_center(event.xdata, event.ydata)
        self.set_radius(0.)

        self._xi = event.xdata
        self._yi = event.ydata

        self._mid_selection = True

    def update_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._mid_selection:
            return

        dx = event.xdata - self._xi
        dy = (event.ydata - self._yi) * aspect_ratio(self._ax)

        self.set_radius(np.sqrt(dx * dx + dy * dy))

    def finalize_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._mid_selection:
            return

        subset = self.get_subset()
        if subset is None: return

        x = self._data.components[self._component_x].data
        y = self._data.components[self._component_y].data

        subset.mask = self.contains(x, y)

        self.reset()

        self._mid_selection = False

class MplTreeTool(AbstractMplRoiTool):
    def __init__(self, data, ax):

        # feed in dummy components -- they aren't needed
        c = data.components.keys()
        AbstractMplRoiTool.__init__(self, data, c[0], c[0], ax)

    def check_compatible_subset(self, subset):
        return isinstance(subset, cv.subset.TreeSubset)

    def start_selection(self, event):
        if not (event.inaxes and self._active):
            return

        self._mid_selection = True

    def update_selection(self, event):
        if not (event.inaxes and self._active and self._mid_selection):
            return

        subset = self.get_subset()
        if subset is None: return

        x = int(event.xdata)
        y = int(event.ydata)
        
        #only works for 2D data!
        assert( len(subset.data.shape) == 2)
        im = subset.data['INDEX_MAP'].data
        index = im[y][x]

        id = subset.data.tree._index[index].get_subtree_indices()
        subset.node_list = id
        
    
    def finalize_selection(self, event):
        if not (event.inaxes and self._active and self._mid_selection):
            return
        self.update_selection(event)
        
        self._mid_selection = False
        
                         
class MplPolygonTool(MplPolygonalROI, AbstractMplRoiTool):

    def __init__(self, data, component_x, component_y, ax):

        MplPolygonalROI.__init__(self, ax)
        AbstractMplRoiTool.__init__(self, data, component_x, component_y, ax)

        self.vx = [0.]
        self.vy = [0.]
        
    def check_compatible_subset(self, subset):
        return isinstance(subset, cv.subset.ElementSubset)

    def add_vertex(self, event):

        if not (event.inaxes and self._active):
            return

        if event.button != 1:
            return

        self.add_point(event.xdata, event.ydata)

        self._mid_selection = True

    def update_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._mid_selection:
            return

        self.replace_last_point(event.xdata, event.ydata)

    def finalize_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._mid_selection:
            return

        if event.button != 3:
            return

        subset = self.get_subset()
        if subset is None: return

        x = self._data.components[self._component_x].data
        y = self._data.components[self._component_y].data

        subset.mask = self.contains(x, y)

        self.reset()

        self.vx = [0.]
        self.vy = [0.]

        self._mid_selection = False


class MplLassoTool(MplPolygonalROI, AbstractMplRoiTool):

    def __init__(self, data, component_x, component_y, ax):

        MplPolygonalROI.__init__(self, ax)
        AbstractMplRoiTool.__init__(self, data, component_x, component_y, ax)

    def check_compatible_subset(self, subset):
        return isinstance(subset, cv.subset.ElementSubset)

    def start_selection(self, event):

        if not (event.inaxes and self._active):
            return

        self.reset()
        self.add_point(event.xdata, event.ydata)

        self._mid_selection = True

    def update_selection(self, event):

        if not (event.inaxes and self._active):
            return

        if not self._mid_selection:
            return

        self.add_point(event.xdata, event.ydata)

    def finalize_selection(self, event):

        if self._polygon is None:
            return

        if not self._mid_selection:
            return

        subset = self.get_subset()
        if subset is None: return

        x = self._data.components[self._component_x].data
        y = self._data.components[self._component_y].data

        subset.mask = self.contains(x, y)

        self.reset()

        self._mid_selection = False
