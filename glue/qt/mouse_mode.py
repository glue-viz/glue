"""MouseModes define various mouse gestures.

The :class:`~glue.qt.glue_toolbar.GlueToolbar` maintains a list of
MouseModes from the visualization it is assigned to, and sees to it
that only one MouseMode is active at a time.

Each MouseMode appears as an Icon in the GlueToolbar. Classes can
assign methods to the press_callback, move_callback, and
release_callback methods of each Mouse Mode, to implement custom
functionality

The basic usage pattern is thus:
 * visualization object instantiates the MouseModes it wants
 * each of these is passed to the add_mode method of the GlueToolbar
 * visualization object optionally attaches methods to the 3 _callback
   methods in a MouseMode, for additional behavior

"""
from PyQt4.QtGui import QIcon, QAction

import numpy as np

from ..core import util
from ..core import roi


class MouseMode(object):
    """ The base class for all MouseModes.

    MouseModes have the following attributes:

    * Icon : QIcon object
    * action_text : The action title (used in some menus)
    * tool_tip : string giving the tool itp
    * shortcut : Keyboard shortcut to toggle the mode
    * _press_callback : Callback method that will be called
      whenever a MouseMode processes a mouse press event
    * _move_callback : Same as above, for move events
    * _release_callback : Same as above, for release events

    The _callback hooks are called with the MouseMode as its only
    argument
    """
    def __init__(self, axes,
                 press_callback=None,
                 move_callback=None,
                 release_callback=None):

        self.icon = None
        self.mode_id = None
        self.action_text = None
        self.tool_tip = None
        self._axes = axes
        self._press_callback = press_callback
        self._move_callback = move_callback
        self._release_callback = release_callback
        self.shortcut = None
        self._event_x = None
        self._event_y = None
        self._event_xdata = None
        self._event_ydata = None

    def _log_position(self, event):
        if event is None:
            return
        self._event_x, self._event_y = event.x, event.y
        self._event_xdata, self._event_ydata = event.xdata, event.ydata

    def press(self, event):
        """ Handles mouse presses

        Logs mouse position and calls press_callback method

        :param event: Mouse event
        :type event: Matplotlib event
        """
        self._log_position(event)
        if self._press_callback is not None:
            self._press_callback(self)

    def move(self, event):
        """ Handles mouse move events

        Logs mouse position and calls move_callback method

        :param event: Mouse event
        :type event: Matplotlib event
        """
        self._log_position(event)
        if self._move_callback is not None:
            self._move_callback(self)

    def release(self, event):
        """ Handles mouse release events.

        Logs mouse position and calls release_callback method

        :param event: Mouse event
        :type event: Matplotlib event
        """
        self._log_position(event)
        if self._release_callback is not None:
            self._release_callback(self)

    def menu_actions(self):
        """ List of QActions to be attached to this mode as a context menu """
        return []


class RoiMode(MouseMode):
    """ Defines ROIs, accessible via the roi() method.

    This is an abstract base class. Subclasses assign an RoiTool
    to the _roi_tool attribute
    """
    def __init__(self, axes, **kwargs):
        super(RoiMode, self).__init__(axes, **kwargs)
        self._roi_tool = None

    def roi(self):
        """ The ROI defined by this mouse mode

        :rtype: :class:`~glue.core.roi.Roi`
        """
        return self._roi_tool.roi()

    def press(self, event):
        self._roi_tool.start_selection(event)
        super(RoiMode, self).press(event)

    def move(self, event):
        self._roi_tool.update_selection(event)
        super(RoiMode, self).move(event)

    def release(self, event):
        self._roi_tool.finalize_selection(event)
        super(RoiMode, self).release(event)


class RectangleMode(RoiMode):
    """ Defines a Rectangular ROI, accessible via the roi() method"""
    def __init__(self, axes, **kwargs):
        super(RectangleMode, self).__init__(axes, **kwargs)
        self.icon = QIcon(':icons/glue_square.png')
        self.mode_id = 'Rectangle'
        self.action_text = 'Rectangular ROI'
        self.tool_tip = 'Define a rectangular region of interest'
        self._roi_tool = roi.MplRectangularROI(self._axes)
        self.shortcut = 'R'


class CircleMode(RoiMode):
    """ Defines a Circular ROI, accessible via the roi() method"""
    def __init__(self, axes, **kwargs):
        super(CircleMode, self).__init__(axes, **kwargs)
        self.icon = QIcon(':icons/glue_circle.png')
        self.mode_id = 'Circle'
        self.action_text = 'Circular ROI'
        self.tool_tip = 'Define a circular region of interest'
        self._roi_tool = roi.MplCircularROI(self._axes)
        self.shortcut = 'C'


class PolyMode(RoiMode):
    """ Defines a Polygonal ROI, accessible via the roi() method"""
    def __init__(self, axes, **kwargs):
        super(PolyMode, self).__init__(axes, **kwargs)
        self.icon = QIcon(':icons/glue_lasso.png')
        self.mode_id = 'Lasso'
        self.action_text = 'Polygonal ROI'
        self.tool_tip = 'Lasso a region of interest'
        self._roi_tool = roi.MplPolygonalROI(self._axes)
        self.shortcut = 'L'


class ContrastMode(MouseMode):
    """Uses right mouse button drags to set bias and contrast, ala DS9

    The horizontal position of the mouse sets the bias, the vertical
    position sets the contrast. The get_scaling method converts
    this information into scaling information for a particular data set
    """
    def __init__(self, *args, **kwargs):
        super(ContrastMode, self).__init__(*args, **kwargs)
        self.icon = QIcon(':icons/glue_contrast.png')
        self.mode_id = 'Contrast'
        self.action_text = 'Contrast'
        self.tool_tip = 'Adjust the bias/contrast'
        self.shortcut = 'B'

        self.bias = 0.5
        self.contrast = 0.5

        self._last = None
        self._result = None
        self._percent_lo = 1.
        self._percent_hi = 99.

    def get_scaling(self, data):
        """ Return the intensity values to set as the darkest and
        lightest color, given the bias and contrast.

        :param data: Raw intensities to scale
        :type data: ndarray

        Returns
           * tuple of lo,hi : the intensity values to set as darkest/brightest
        :rtype: tuple
        """
        lo, hi = self.get_bounds(data)
        ra = hi - lo
        bias = lo + ra * self.bias
        vmin = bias - ra * self.contrast
        vmax = bias + ra * self.contrast
        return vmin, vmax

    def _downsample(self, data):
        shp = data.shape
        slices = tuple(slice(None, None, max(1, s / 100)) for s in shp)
        return data[slices]

    def get_bounds(self, data):
        #cache last result. cant use @memoize, since ndarrays dont hash
        #XXX warning -- results are bad if data values change in-place
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("Contrast MouseMode requires SciPy")
        if data is not self._last:
            self._last = data
            limits = (-np.inf, np.inf)
            d = self._downsample(data)
            lo = stats.scoreatpercentile(d.flat, self._percent_lo,
                                         limit=limits)
            hi = stats.scoreatpercentile(d.flat, self._percent_hi,
                                         limit=limits)
            self._result = lo, hi
        return self._result

    def set_clip_percentile(self, lo, hi):
        """Percentiles at which to clip the data at black/white"""
        if lo == self._percent_lo and hi == self._percent_hi:
            return
        self._percent_lo = lo
        self._percent_hi = hi
        self._last = None  # clear cache

    def move(self, event):
        """ MoveEvent. Update bias and contrast on Right Mouse button drag """
        if event.button != 3:  # RMB drag only
            return
        x, y = event.x, event.y
        dx, dy = self._axes.figure.canvas.get_width_height()
        x = 1.0 * x / dx
        y = 1.0 * y / dy
        theta = np.pi - max(min(y, 1), 0) * np.pi

        self.bias = x
        self.contrast = np.tan(theta)

        super(ContrastMode, self).move(event)

    def menu_actions(self):
        result = []

        a = QAction("minmax", None)
        a.triggered.connect(lambda: self.set_clip_percentile(0, 100))
        result.append(a)

        a = QAction("99%", None)
        a.triggered.connect(lambda: self.set_clip_percentile(1, 99))
        result.append(a)

        a = QAction("95%", None)
        a.triggered.connect(lambda: self.set_clip_percentile(5, 95))
        result.append(a)

        a = QAction("90%", None)
        a.triggered.connect(lambda: self.set_clip_percentile(10, 90))
        result.append(a)

        for r in result:
            if self._move_callback is not None:
                r.triggered.connect(lambda: self._move_callback(self))
        return result


class ContourMode(MouseMode):
    """ Creates ROIs by using the mouse to 'pick' contours out of the data """
    def __init__(self, *args, **kwargs):
        super(ContourMode, self).__init__(*args, **kwargs)

        self.icon = QIcon(":icons/glue_contour.png")
        self.mode_id = 'Contour'
        self.action_text = 'Contour'
        self.tool_tip = 'Define a region of intrest via contours'
        self.shortcut = 'N'

    def roi(self, data):
        """Caculate an ROI as the contour which passes through the mouse

        :param data: The data set to use
        :type data: ndarray

        Returns

           * A :class:`~glue.core.roi.PolygonalROI` object, or None if one
             could not be calculated

        This method calculates the (single) contour that passes
        through the mouse location, and uses this path to define
        a new ROI
        """
        x, y = self._event_xdata, self._event_ydata
        return contour_to_roi(x, y, data)


def contour_to_roi(x, y, data):
    """ Return a PolygonalROI for the contour that passes through (x,y) in data

    :param x: x coordinate
    :param y: y coordinate
    :param data: data
    :type data: numpy array

    Returns:
       * A :class:`~glue.core.roi.PolygonalROI` instance
    """
    if x is None or y is None:
        return None

    xy = util.point_contour(x, y, data)
    if xy is None:
        return None

    p = roi.PolygonalROI(vx=xy[:, 0], vy=xy[:, 1])
    return p
