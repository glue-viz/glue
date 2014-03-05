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
from ..external.qt.QtGui import QAction

import numpy as np

from ..core import util
from ..core import roi
from . import get_qapp
from .qtutil import get_icon, nonpartial
from . import qt_roi


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
                 release_callback=None,
                 key_callback=None):

        self.icon = None
        self.mode_id = None
        self.action_text = None
        self.tool_tip = None
        self._axes = axes
        self._press_callback = press_callback
        self._move_callback = move_callback
        self._release_callback = release_callback
        self._key_callback = key_callback
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

    def key(self, event):
        """ Handles key press events

        Calls key_callback method

        :param event: Key event
        :type event: Matplotlib event
        """
        if self._key_callback is not None:
            self._key_callback(self)

    def menu_actions(self):
        """ List of QActions to be attached to this mode as a context menu """
        return []


class RoiModeBase(MouseMode):

    """ Base class for defining ROIs. ROIs accessible via the roi() method

    See RoiMode and ClickRoiMode subclasses for interaction details

    Clients can provide an roi_callback function. When ROIs are
    finalized (i.e. fully defined), this function will be called with
    the RoiMode object as the argument. Clients can use RoiMode.roi()
    to retrieve the new ROI, and take the appropriate action.
    """

    def __init__(self, axes, **kwargs):
        """
        :param roi_callback: Function that will be called when the
                             ROI is finished being defined.
        :type roi_callback:  function
        """
        self._roi_callback = kwargs.pop('roi_callback', None)
        super(RoiModeBase, self).__init__(axes, **kwargs)
        self._roi_tool = None

    def roi(self):
        """ The ROI defined by this mouse mode

        :rtype: :class:`~glue.core.roi.Roi`
        """
        return self._roi_tool.roi()

    def _finish_roi(self, event):
        """Called by subclasses when ROI is fully defined"""
        self._roi_tool.finalize_selection(event)
        if self._roi_callback is not None:
            self._roi_callback(self)

class RoiMode(RoiModeBase):

    """ Define Roi Modes via click+drag events

    ROIs are updated continuously on click+drag events, and finalized
    on each mouse release
    """

    def __init__(self, axes, **kwargs):
        super(RoiMode, self).__init__(axes, **kwargs)

        self._start_event = None
        self._drag = False
        app = get_qapp()
        self._drag_dist = app.startDragDistance()

    def _update_drag(self, event):
        if self._drag or self._start_event is None:
            return

        dx = abs(event.x - self._start_event.x)
        dy = abs(event.y - self._start_event.y)
        if (dx + dy) > self._drag_dist:
            self._roi_tool.start_selection(self._start_event)
            self._drag = True

    def press(self, event):
        self._start_event = event
        super(RoiMode, self).press(event)

    def move(self, event):
        self._update_drag(event)
        if self._drag:
            self._roi_tool.update_selection(event)
        super(RoiMode, self).move(event)

    def release(self, event):
        if self._drag:
            self._finish_roi(event)
        self._drag = False
        self._start_event = None

        super(RoiMode, self).release(event)


class PersistentRoiMode(RoiMode):
    """
    Same functionality as RoiMode, but the Roi is never
    finalized, and remains rendered after mouse gestures
    """
    def _finish_roi(self, event):
        if self._roi_callback is not None:
            self._roi_callback(self)


class ClickRoiMode(RoiModeBase):

    """
    Generate ROIs using clicks and click+drags.

    ROIs updated on each click, and each click+drag.
    ROIs are finalized on enter press, and reset on escape press
    """

    def __init__(self, axes, **kwargs):
        super(ClickRoiMode, self).__init__(axes, **kwargs)
        self._last_event = None

    def press(self, event):
        if not self._roi_tool.active():
            self._roi_tool.start_selection(event)
        else:
            self._roi_tool.update_selection(event)
        self._last_event = event
        super(ClickRoiMode, self).press(event)

    def move(self, event):
        if event.button is not None and self._roi_tool.active():
            self._roi_tool.update_selection(event)
            self._last_event = event
        super(ClickRoiMode, self).move(event)

    def key(self, event):
        if event.key == 'enter':
            self._finish_roi(self._last_event)
        elif event.key == 'escape':
            self._roi_tool.reset()
        super(ClickRoiMode, self).key(event)


class RectangleMode(RoiMode):
    """ Defines a Rectangular ROI, accessible via the roi() method"""

    def __init__(self, axes, **kwargs):
        super(RectangleMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_square')
        self.mode_id = 'Rectangle'
        self.action_text = 'Rectangular ROI'
        self.tool_tip = 'Define a rectangular region of interest'
        self._roi_tool = qt_roi.QtRectangularROI(self._axes)
        self.shortcut = 'R'


class CircleMode(RoiMode):

    """ Defines a Circular ROI, accessible via the roi() method"""

    def __init__(self, axes, **kwargs):
        super(CircleMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_circle')
        self.mode_id = 'Circle'
        self.action_text = 'Circular ROI'
        self.tool_tip = 'Define a circular region of interest'
        self._roi_tool = qt_roi.QtCircularROI(self._axes)
        self.shortcut = 'C'


class PolyMode(ClickRoiMode):

    """ Defines a Polygonal ROI, accessible via the roi() method"""

    def __init__(self, axes, **kwargs):
        super(PolyMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_lasso')
        self.mode_id = 'Polygon'
        self.action_text = 'Polygonal ROI'
        self.tool_tip = 'Lasso a region of interest'
        self._roi_tool = qt_roi.QtPolygonalROI(self._axes)
        self.shortcut = 'G'


class LassoMode(RoiMode):

    """ Defines a Polygonal ROI, accessible via the roi() method"""

    def __init__(self, axes, **kwargs):
        super(LassoMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_lasso')
        self.mode_id = 'Lasso'
        self.action_text = 'Polygonal ROI'
        self.tool_tip = 'Lasso a region of interest'
        self._roi_tool = qt_roi.QtPolygonalROI(self._axes)
        self.shortcut = 'L'


class HRangeMode(RoiMode):

    """ Defines a Range ROI, accessible via the roi() method.
    This class defines horizontal ranges"""

    def __init__(self, axes, **kwargs):
        super(HRangeMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_xrange_select')
        self.mode_id = 'X range'
        self.action_text = 'X range'
        self.tool_tip = 'Select a range of x values'
        self._roi_tool = qt_roi.QtXRangeROI(self._axes)
        self.shortcut = 'H'


class VRangeMode(RoiMode):

    """ Defines a Range ROI, accessible via the roi() method.
    This class defines vertical ranges"""

    def __init__(self, axes, **kwargs):
        super(VRangeMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_yrange_select')
        self.mode_id = 'Y range'
        self.action_text = 'Y range'
        self.tool_tip = 'Select a range of y values'
        self._roi_tool = qt_roi.QtYRangeROI(self._axes)
        self.shortcut = 'V'


class ContrastMode(MouseMode):

    """Uses right mouse button drags to set bias and contrast, ala DS9

    The horizontal position of the mouse sets the bias, the vertical
    position sets the contrast. The get_scaling method converts
    this information into scaling information for a particular data set
    """

    def __init__(self, *args, **kwargs):
        super(ContrastMode, self).__init__(*args, **kwargs)
        self.icon = get_icon('glue_contrast')
        self.mode_id = 'Contrast'
        self.action_text = 'Contrast'
        self.tool_tip = 'Adjust the bias/contrast'
        self.shortcut = 'B'

        self.bias = 0.5
        self.contrast = 1.0

        self._last = None
        self._result = None
        self._percent_lo = 1.
        self._percent_hi = 99.
        self.stretch = 'linear'

    def set_clip_percentile(self, lo, hi):
        """Percentiles at which to clip the data at black/white"""
        if lo == self._percent_lo and hi == self._percent_hi:
            return
        self._percent_lo = lo
        self._percent_hi = hi

    def get_clip_percentile(self):
        return self._percent_lo, self._percent_hi

    def move(self, event):
        """ MoveEvent. Update bias and contrast on Right Mouse button drag """
        if event.button != 3:  # RMB drag only
            return
        x, y = event.x, event.y
        dx, dy = self._axes.figure.canvas.get_width_height()
        x = 1.0 * x / dx
        y = 1.0 * y / dy

        self.bias = x
        self.contrast = (1 - y) * 10

        super(ContrastMode, self).move(event)

    def menu_actions(self):
        result = []

        a = QAction("minmax", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 0, 100))
        result.append(a)

        a = QAction("99%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 1, 99))
        result.append(a)

        a = QAction("95%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 5, 95))
        result.append(a)

        a = QAction("90%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 10, 90))
        result.append(a)

        a = QAction("", None)
        a.setSeparator(True)
        result.append(a)

        a = QAction("linear", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'linear'))
        result.append(a)

        a = QAction("log", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'log'))
        result.append(a)

        a = QAction("power", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'power'))
        result.append(a)

        a = QAction("square root", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'sqrt'))
        result.append(a)

        a = QAction("squared", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'squared'))
        result.append(a)

        a = QAction("asinh", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'arcsinh'))
        result.append(a)

        for r in result:
            if self._move_callback is not None:
                r.triggered.connect(nonpartial(self._move_callback, self))

        return result


class SpectrumExtractorMode(PersistentRoiMode):
    """
    Let's the user select a region in an image and,
    when connected to a SpectrumExtractorTool, uses this
    to display spectra extracted from that position
    """
    def __init__(self, axes, **kwargs):
        super(SpectrumExtractorMode, self).__init__(axes, **kwargs)
        self.icon = get_icon('glue_square')
        self.mode_id = 'Spectrum'
        self.action_text = 'Extract a spectrum from a selection'
        self._roi_tool = qt_roi.QtRectangularROI(self._axes)
        self._roi_tool.plot_opts.update(edgecolor='#91cf60',
                                        facecolor=None,
                                        edgewidth=3,
                                        alpha=1.0)
        self.shortcut = 'S'


class ContourMode(MouseMode):

    """ Creates ROIs by using the mouse to 'pick' contours out of the data """

    def __init__(self, *args, **kwargs):
        super(ContourMode, self).__init__(*args, **kwargs)

        self.icon = get_icon("glue_contour")
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
