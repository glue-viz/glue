"""MouseModes define various mouse gestures.

The :class:`~glue.viewers.common.qt.toolbar.GlueToolbar` maintains a list of
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

from __future__ import absolute_import, division, print_function

import os

from qtpy import QtGui, QtWidgets
from glue.core.callback_property import CallbackProperty
from glue.core import roi
from glue.core.qt import roi as qt_roi
from glue.utils.qt import get_qapp
from glue.icons.qt import get_icon
from glue.utils import nonpartial
from glue.utils.qt import load_ui
from glue.viewers.common.qt.mode import CheckableMode
from glue.config import toolbar_mode


class MouseMode(CheckableMode):
    """
    The base class for all MouseModes.

    MouseModes have the following attributes:

    * press_callback : Callback method that will be called
      whenever a MouseMode processes a mouse press event
    * move_callback : Same as above, for move events
    * release_callback : Same as above, for release events
    * key_callback : Same as above, for release events

    The _callback hooks are called with the MouseMode as its only
    argument
    """
    enabled = CallbackProperty(True)

    def __init__(self, viewer,
                 press_callback=None,
                 move_callback=None,
                 release_callback=None,
                 key_callback=None):

        self.viewer = viewer
        self._axes = viewer.axes
        self._press_callback = press_callback
        self._move_callback = move_callback
        self._release_callback = release_callback
        self._key_callback = key_callback
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
        """
        Handles mouse presses

        Logs mouse position and calls press_callback method

        :param event: Mouse event
        :type event: Matplotlib event
        """
        self._log_position(event)
        if self._press_callback is not None:
            self._press_callback(self)

    def move(self, event):
        """
        Handles mouse move events

        Logs mouse position and calls move_callback method

        :param event: Mouse event
        :type event: Matplotlib event
        """
        self._log_position(event)
        if self._move_callback is not None:
            self._move_callback(self)

    def release(self, event):
        """
        Handles mouse release events.

        Logs mouse position and calls release_callback method

        :param event: Mouse event
        :type event: Matplotlib event
        """
        self._log_position(event)
        if self._release_callback is not None:
            self._release_callback(self)

    def key(self, event):
        """
        Handles key press events

        Calls key_callback method

        :param event: Key event
        :type event: Matplotlib event
        """
        if self._key_callback is not None:
            self._key_callback(self)


class RoiModeBase(MouseMode):

    """ Base class for defining ROIs. ROIs accessible via the roi() method

    See RoiMode and ClickRoiMode subclasses for interaction details

    Clients can provide an roi_callback function. When ROIs are
    finalized (i.e. fully defined), this function will be called with
    the RoiMode object as the argument. Clients can use RoiMode.roi()
    to retrieve the new ROI, and take the appropriate action.
    """
    persistent = False  # clear the shape when drawing completes?

    def __init__(self, viewer, **kwargs):
        """
        :param roi_callback: Function that will be called when the
                             ROI is finished being defined.
        :type roi_callback:  function
        """
        self._roi_callback = kwargs.pop('roi_callback', None)
        super(RoiModeBase, self).__init__(viewer, **kwargs)
        self._roi_tool = None

    def activate(self):
        self._roi_tool._sync_patch()

    def roi(self):
        """ The ROI defined by this mouse mode

        :rtype: :class:`~glue.core.roi.Roi`
        """
        return self._roi_tool.roi()

    def _finish_roi(self, event):
        """Called by subclasses when ROI is fully defined"""
        if not self.persistent:
            self._roi_tool.finalize_selection(event)
        if self._roi_callback is not None:
            self._roi_callback(self)

    def clear(self):
        self._roi_tool.reset()


class RoiMode(RoiModeBase):

    """ Define Roi Modes via click+drag events

    ROIs are updated continuously on click+drag events, and finalized
    on each mouse release
    """

    def __init__(self, viewer, **kwargs):
        super(RoiMode, self).__init__(viewer, **kwargs)

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
            status = self._roi_tool.start_selection(self._start_event)
            # If start_selection returns False, the selection has not been
            # started and we should abort, so we set self._drag to False in this
            # case.
            self._drag = True if status is None else status

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

    def key(self, event):
        if event.key == 'escape':
            self._roi_tool.abort_selection(event)
            self._drag = False
            self._drawing = False
            self._start_event = None
        super(RoiMode, self).key(event)


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

    def __init__(self, viewer, **kwargs):
        super(ClickRoiMode, self).__init__(viewer, **kwargs)
        self._last_event = None
        self._drawing = False

    def press(self, event):
        if not self._roi_tool.active() or not self._drawing:
            self._roi_tool.start_selection(event)
            self._drawing = True
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
            self._drawing = False
        elif event.key == 'escape':
            self._roi_tool.abort_selection(event)
            self._drawing = False
        super(ClickRoiMode, self).key(event)

    def release(self, event):
        if getattr(self._roi_tool, '_scrubbing', False):
            self._finish_roi(event)
            self._start_event = None
            super(ClickRoiMode, self).release(event)


@toolbar_mode
class RectangleMode(RoiMode):

    """ Defines a Rectangular ROI, accessible via the roi() method"""

    icon = 'glue_square'
    mode_id = 'Rectangle'
    action_text = 'Rectangular ROI'
    tool_tip = 'Define a rectangular region of interest'
    shortcut = 'R'

    def __init__(self, viewer, **kwargs):
        super(RectangleMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtRectangularROI(self._axes)


class PathMode(ClickRoiMode):

    persistent = True

    def __init__(self, viewer, **kwargs):
        super(PathMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtPathROI(self._axes)

        self._roi_tool.plot_opts.update(edgecolor='#de2d26',
                                        facecolor=None,
                                        edgewidth=3,
                                        alpha=0.4)


@toolbar_mode
class CircleMode(RoiMode):
    """
    Defines a Circular ROI, accessible via the roi() method
    """

    icon = 'glue_circle'
    mode_id = 'Circle'
    action_text = 'Circular ROI'
    tool_tip = 'Define a circular region of interest'
    shortcut = 'C'

    def __init__(self, viewer, **kwargs):
        super(CircleMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtCircularROI(self._axes)


@toolbar_mode
class PolyMode(ClickRoiMode):
    """
    Defines a Polygonal ROI, accessible via the roi() method
    """

    icon = 'glue_lasso'
    mode_id = 'Polygon'
    action_text = 'Polygonal ROI'
    tool_tip = ('Lasso a region of interest\n'
                '  ENTER accepts the path\n'
                '  ESCAPE clears the path')
    shortcut = 'G'

    def __init__(self, viewer, **kwargs):
        super(PolyMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtPolygonalROI(self._axes)


@toolbar_mode
class LassoMode(RoiMode):
    """
    Defines a Polygonal ROI, accessible via the roi() method
    """

    icon = 'glue_lasso'
    mode_id = 'Lasso'
    action_text = 'Polygonal ROI'
    tool_tip = 'Lasso a region of interest'
    shortcut = 'L'

    def __init__(self, viewer, **kwargs):
        super(LassoMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtPolygonalROI(self._axes)


@toolbar_mode
class HRangeMode(RoiMode):
    """
    Defines a Range ROI, accessible via the roi() method.
    This class defines horizontal ranges
    """

    icon = 'glue_xrange_select'
    mode_id = 'X range'
    action_text = 'X range'
    tool_tip = 'Select a range of x values'
    shortcut = 'H'

    def __init__(self, viewer, **kwargs):
        super(HRangeMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtXRangeROI(self._axes)


@toolbar_mode
class VRangeMode(RoiMode):
    """
    Defines a Range ROI, accessible via the roi() method.
    This class defines vertical ranges
    """

    icon = 'glue_yrange_select'
    mode_id = 'Y range'
    action_text = 'Y range'
    tool_tip = 'Select a range of y values'
    shortcut = 'V'

    def __init__(self, viewer, **kwargs):
        super(VRangeMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtYRangeROI(self._axes)


@toolbar_mode
class PickMode(RoiMode):
    """
    Defines a PointROI. Defines single point selections
    """

    icon = 'glue_yrange_select'
    mode_id = 'Pick'
    action_text = 'Pick'
    tool_tip = 'Select a single item'
    shortcut = 'K'

    def __init__(self, viewer, **kwargs):
        super(PickMode, self).__init__(viewer, **kwargs)
        self._roi_tool = roi.MplPickROI(self._axes)

    def press(self, event):
        super(PickMode, self).press(event)
        self._drag = True


@toolbar_mode
class ContrastMode(MouseMode):
    """
    Uses right mouse button drags to set bias and contrast, ala DS9

    The horizontal position of the mouse sets the bias, the vertical
    position sets the contrast. The get_scaling method converts
    this information into scaling information for a particular data set
    """

    icon = 'glue_contrast'
    mode_id = 'Contrast'
    saction_text = 'Contrast'
    tool_tip = 'Adjust the bias/contrast'
    shortcut = 'B'

    def __init__(self, viewer, **kwargs):
        super(ContrastMode, self).__init__(viewer, **kwargs)

        self.bias = 0.5
        self.contrast = 1.0

        self._last = None
        self._result = None
        self._percent_lo = 1.
        self._percent_hi = 99.
        self.stretch = 'linear'
        self._vmin = None
        self._vmax = None

    def set_clip_percentile(self, lo, hi):
        """Percentiles at which to clip the data at black/white"""
        if lo == self._percent_lo and hi == self._percent_hi:
            return
        self._percent_lo = lo
        self._percent_hi = hi
        self._vmin = None
        self._vmax = None

    def get_clip_percentile(self):
        if self._vmin is None and self._vmax is None:
            return self._percent_lo, self._percent_hi
        return None, None

    def get_vmin_vmax(self):
        if self._percent_lo is None or self._percent_hi is None:
            return self._vmin, self._vmax
        return None, None

    def set_vmin_vmax(self, vmin, vmax):
        if vmin == self._vmin and vmax == self._vmax:
            return
        self._percent_hi = self._percent_lo = None
        self._vmin = vmin
        self._vmax = vmax

    def choose_vmin_vmax(self):
        dialog = load_ui('contrastlimits.ui', None,
                         directory=os.path.dirname(__file__))
        v = QtGui.QDoubleValidator()
        dialog.vmin.setValidator(v)
        dialog.vmax.setValidator(v)

        vmin, vmax = self.get_vmin_vmax()
        if vmin is not None:
            dialog.vmin.setText(str(vmin))
        if vmax is not None:
            dialog.vmax.setText(str(vmax))

        def _apply():
            try:
                vmin = float(dialog.vmin.text())
                vmax = float(dialog.vmax.text())
                self.set_vmin_vmax(vmin, vmax)
                if self._move_callback is not None:
                    self._move_callback(self)
            except ValueError:
                pass

        bb = dialog.buttonBox
        bb.button(bb.Apply).clicked.connect(_apply)
        dialog.accepted.connect(_apply)
        dialog.show()
        dialog.raise_()
        dialog.exec_()

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

        a = QtWidgets.QAction("minmax", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 0, 100))
        result.append(a)

        a = QtWidgets.QAction("99%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 1, 99))
        result.append(a)

        a = QtWidgets.QAction("95%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 5, 95))
        result.append(a)

        a = QtWidgets.QAction("90%", None)
        a.triggered.connect(nonpartial(self.set_clip_percentile, 10, 90))
        result.append(a)

        rng = QtWidgets.QAction("Set range...", None)
        rng.triggered.connect(nonpartial(self.choose_vmin_vmax))
        result.append(rng)

        a = QtWidgets.QAction("", None)
        a.setSeparator(True)
        result.append(a)

        a = QtWidgets.QAction("linear", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'linear'))
        result.append(a)

        a = QtWidgets.QAction("log", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'log'))
        result.append(a)

        a = QtWidgets.QAction("power", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'power'))
        result.append(a)

        a = QtWidgets.QAction("square root", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'sqrt'))
        result.append(a)

        a = QtWidgets.QAction("squared", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'squared'))
        result.append(a)

        a = QtWidgets.QAction("asinh", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'arcsinh'))
        result.append(a)

        for r in result:
            if r is rng:
                continue
            if self._move_callback is not None:
                r.triggered.connect(nonpartial(self._move_callback, self))

        return result
