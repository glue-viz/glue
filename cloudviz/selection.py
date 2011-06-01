import time

import numpy as np
np.seterr(all='ignore')
from matplotlib.patches import Rectangle, Polygon


def point_in_polygon(vx, vy, x, y):
    '''
    Check if a point (x, y) is inside a polygon (vx, vy)
    Polygon vertices have to be given as Numpy arrays
    '''
    xi = vx
    xj = np.roll(vx, 1)
    yi = vy
    yj = np.roll(vy, 1)
    c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
    return np.sum(c) % 2 == 1


class Selection(object):

    def __init__(self, ax, points, subset):
        self.ax = ax
        self.points = points
        self.subset = subset
        self.subset.mask = np.zeros(points.get_offsets().shape[0])

    def refresh(self):
        self.ax.figure.canvas.draw()


class BasePolygonSelection(Selection):

    def select(self):

        # Add first points at the end of the list of vertices to close polygon
        self.x.append(self.x[0])
        self.y.append(self.y[0])

        # Update the vertices
        self.polygon.set_xy(np.array(zip(self.x, self.y)))

        # Refresh the display
        self.refresh()

        # Get the positions of the datapoints
        positions = self.points.get_offsets()

        # Compute mask of which points were selected
        mask = []
        for i in range(positions.shape[0]):
            mask.append(point_in_polygon(self.x, self.y,
                                         positions[i, 0], positions[i, 1]))
        self.subset.mask = np.array(mask, dtype=bool)

        # Change the color of the selected points
        facecolors = ['red' if x else 'green' for x in mask]
        self.points.set_facecolors(facecolors)
        self.refresh()

        # Get rid of the polygon after a small delay
        time.sleep(0.2)
        self.ax.patches.remove(self.polygon)
        self.polygon = None
        self.refresh()


class RectangleSelection(Selection):

    def __init__(self, ax, points, subset):

        Selection.__init__(self, ax, points, subset)

        self.box = None

        self.ax.figure.canvas.mpl_connect('button_press_event',
                                          self.start_selection)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.update_selection)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.finalize_selection)

    def start_selection(self, event, **kwargs):

        if not event.inaxes:
            return

        self.xmin = event.xdata
        self.xmax = event.xdata
        self.ymin = event.ydata
        self.ymax = event.ydata

        self.box = Rectangle((self.xmin, self.xmax),
                             self.xmax - self.xmin,
                             self.ymax - self.ymin,
                             edgecolor='red', facecolor='none', alpha=0.3)

        self.ax.add_patch(self.box)
        self.refresh()

    def update_selection(self, event, **kwargs):

        if self.box is None:
            return

        if not event.inaxes:
            return

        self.xmin = min(event.xdata, self.xmin)
        self.xmax = max(event.xdata, self.xmax)
        self.ymin = min(event.ydata, self.ymin)
        self.ymax = max(event.ydata, self.ymax)

        self.box.set_xy((self.xmin, self.ymin))
        self.box.set_width(self.xmax - self.xmin)
        self.box.set_height(self.ymax - self.ymin)
        self.refresh()

    def finalize_selection(self, event, **kwargs):

        if self.box is None:
            return

        # Get the positions of the datapoints
        positions = self.points.get_offsets()

        # Compute mask of which points were selected
        mask = []
        for i in range(positions.shape[0]):
            mask.append(positions[i, 0] > self.xmin and \
                        positions[i, 0] < self.xmax and \
                        positions[i, 1] > self.ymin and \
                        positions[i, 1] < self.ymax)
        self.subset.mask = np.array(mask, dtype=bool)

        # Change the color of the selected points
        facecolors = ['red' if x else 'green' for x in mask]
        self.points.set_facecolors(facecolors)
        self.refresh()

        # Get rid of the box after a small delay
        time.sleep(0.2)
        self.ax.patches.remove(self.box)
        self.box = None
        self.refresh()


class LassoSelection(BasePolygonSelection):

    def __init__(self, ax, points, subset):

        BasePolygonSelection.__init__(self, ax, points, subset)

        self._initialize_polygon()

        self.ax.figure.canvas.mpl_connect('button_press_event',
                                          self.start_selection)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.update_selection)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.finalize_selection)

    def _initialize_polygon(self):

        self.polygon = None
        self.xpix_last = 0
        self.ypix_last = 0

    def start_selection(self, event, **kwargs):

        if not event.inaxes:
            return

        # When the mouse button is first clicked, add current position and
        # create polygon
        self.x = [event.xdata]
        self.y = [event.ydata]
        self.polygon = Polygon(np.array(zip(self.x, self.y)),
                               edgecolor='red', facecolor='none', alpha=0.3)
        self.ax.add_patch(self.polygon)
        self.refresh()

    def update_selection(self, event, **kwargs):

        if self.polygon is None:
            return

        if not event.inaxes:
            return

        # Only add point if it has changed since last position
        if event.x == self.xpix_last and event.y == self.ypix_last:
            return

        # Update polygon with current position
        self.x.append(event.xdata)
        self.y.append(event.ydata)
        self.polygon.set_xy(np.array(zip(self.x, self.y)))
        self.refresh()

        # Remember current position
        self.xpix_last = event.x
        self.ypix_last = event.y

    def finalize_selection(self, event, **kwargs):

        if self.polygon is None:
            return

        self.select()

        self._initialize_polygon


class PolygonSelection(BasePolygonSelection):

    def __init__(self, ax, points, subset):

        BasePolygonSelection.__init__(self, ax, points, subset)

        self._initialize_polygon()

        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.update_current)
        self.ax.figure.canvas.mpl_connect('button_press_event',
                                          self.add_point)
        self.ax.figure.canvas.mpl_connect('button_press_event',
                                          self.finalize_selection)

    def _initialize_polygon(self):

        self.x = [0.]
        self.y = [0.]
        self.polygon = Polygon(np.array(zip(self.x, self.y)),
                               edgecolor='red', facecolor='none', alpha=0.3)
        self.ax.add_patch(self.polygon)

    def add_point(self, event, **kwargs):

        if not event.inaxes:
            return

        if self.polygon is None:
            return

        if not event.button == 1:
            return

        # Update polygon with current position
        self.x.append(event.xdata)
        self.y.append(event.ydata)
        self.polygon.set_xy(np.array(zip(self.x, self.y)))
        self.refresh()

    def update_current(self, event, **kwargs):

        if not event.inaxes:
            return

        # Draw polygon up to current cursor position
        self.x[-1] = event.xdata
        self.y[-1] = event.ydata
        self.polygon.set_xy(np.array(zip(self.x, self.y)))
        self.refresh()

    def finalize_selection(self, event, **kwargs):

        if self.polygon is None:
            return

        if not event.button == 3:
            return

        self.select()

        self._initialize_polygon()
