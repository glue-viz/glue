import numpy as np
np.seterr(all='ignore')
from matplotlib.patches import Polygon


class Roi(object):
    """
    A class to define 2D regions-of-interest
    """

    def __init__(self):
        """ Create a new ROI """
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

        #special case of empty ROI
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


class MplRoi(Roi):
    """
    A subclass of ROI that also renders the ROI to a plot

    Attributes:
    -----------
    plot_opts: Dictionary instance
               A dictionary of plot keywords that are passed to
               the polygon patch representing the ROI. These control
               the visual properties of the ROI
    """

    def __init__(self, ax):
        """ Create a new roi

        Parameters
        ----------
        ax: A matplotlib Axes object to attach the graphical ROI to
        """

        Roi.__init__(self)
        self.plot_opts = {'edgecolor': 'red', 'facecolor': 'none',
                          'alpha': 0.3}
        self._polygon = Polygon(np.array(zip([0, 1], [0, 1])))
        self._polygon.set(**self.plot_opts)
        self._ax = ax
        self._ax.add_patch(self._polygon)
        self._sync_patch()

    def _sync_patch(self):
        if not self.vx:
            self._polygon.set_visible(False)
        else:
            self._polygon.set_xy(np.array(zip(self.vx + [self.vx[0]],
                                              self.vy + [self.vy[0]])))
            self._polygon.set_visible(True)
        self._polygon.set(**self.plot_opts)
        self._ax.figure.canvas.draw()

    def add_point(self, x, y):
        Roi.add_point(self, x, y)
        self._sync_patch()

    def reset(self):
        Roi.reset(self)
        self._sync_patch()

    def remove_point(self, x, y, thresh=None):
        Roi.reset(self, x, y, thresh=None)
        self._sync_patch()
