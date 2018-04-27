"""
ToolbarModes are CheckableTools that enable various MouseModes.

The toolbar maintains a list of MouseModes from the visualization it is
assigned to, and sees to it that only one MouseMode is active at a time.

Each ToolbarMode appears as an Icon in the toolbar. Classes can assign methods
to the press_callback, move_callback, and release_callback methods of each
Mouse Mode, to implement custom functionality

The basic usage pattern is thus:
 * visualization object instantiates the MouseModes it wants
 * each of these is passed to the add_tool method of the toolbar
 * visualization object optionally attaches methods to the 3 _callback
   methods in a MouseMode, for additional behavior
"""

from __future__ import absolute_import, division, print_function

import os

from qtpy import QtGui, QtWidgets
from glue.core import roi
from glue.core.qt import roi as qt_roi
from glue.utils.qt import get_qapp
from glue.utils import nonpartial
from glue.utils.qt import load_ui, cmap2pixmap
from glue.viewers.common.qt.tool import Tool, CheckableTool
from glue.config import viewer_tool
from .mouse_mode import MouseMode


__all__ = ['ToolbarModeBase', 'RoiModeBase', 'RoiMode', 'PersistentRoiMode',
           'ClickRoiMode', 'RectangleMode', 'PathMode', 'CircleMode',
           'PolyMode', 'LassoMode', 'HRangeMode', 'VRangeMode', 'PickMode',
           'ContrastMode', 'ColormapMode']


class ToolbarModeBase(MouseMode, CheckableTool):
    """
    All ToolbarModes are both MouseModes and CheckableTools
    """

    def __init__(self, viewer, **kwargs):
        MouseMode.__init__(self, viewer, **kwargs)
        CheckableTool.__init__(self, viewer=viewer)


class RoiModeBase(ToolbarModeBase):
    """
    Base class for defining ROIs. ROIs accessible via the roi() method

    See RoiMode and ClickRoiMode subclasses for interaction details

    An roi_callback function can be provided. When ROIs are finalized (i.e.
    fully defined), this function will be called with the RoiMode object as the
    argument. Clients can use RoiMode.roi() to retrieve the new ROI, and take
    the appropriate action. By default, roi_callback will default to calling an
    ``apply_roi`` method on the data viewer.
    """
    persistent = False  # clear the shape when drawing completes?
    trigger_on_finalize = True

    def __init__(self, viewer, **kwargs):
        """
        Parameters
        ----------
        roi_callback : `func`
            Function that will be called when the ROI is finished being
            defined.
        """
        def apply_mode(mode):
            self.viewer.apply_roi(self.roi())
        self._roi_callback = kwargs.pop('roi_callback', apply_mode)
        super(RoiModeBase, self).__init__(viewer, **kwargs)
        self._roi_tool = None

    def close(self, *args):
        self._roi_callback = None
        super(RoiModeBase, self).close()

    def activate(self):
        self._roi_tool._sync_patch()
        super(RoiModeBase, self).activate()

    def roi(self):
        """
        The ROI defined by this mouse mode

        Returns
        -------
        roi : :class:`~glue.core.roi.Roi`
        """
        return self._roi_tool.roi()

    def _finish_roi(self, event):
        """
        Called by subclasses when ROI is fully defined
        """
        if not self.persistent:
            self._roi_tool.finalize_selection(event)
        if self._roi_callback is not None:
            self._roi_callback(self)
        if self.trigger_on_finalize:
            # TODO: there is perhaps a more elegant way to handle this
            self.viewer.toolbar.actions[self.tool_id].trigger()

    def clear(self):
        self._roi_tool.reset()


class RoiMode(RoiModeBase):
    """
    Define Roi Modes via click+drag events.

    ROIs are updated continuously on click+drag events, and finalized on each
    mouse release
    """

    status_tip = "CLICK and DRAG to define selection, CTRL-CLICK and DRAG to move selection"

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
            # started and we should abort, so we set self._drag to False in
            # this case.
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
    ROIs are finalized on enter press, and reset on escape press.
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


@viewer_tool
class RectangleMode(RoiMode):
    """
    Defines a Rectangular ROI, accessible via the :meth:`~RectangleMode.roi`
    method
    """

    icon = 'glue_square'
    tool_id = 'select:rectangle'
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


@viewer_tool
class CircleMode(RoiMode):
    """
    Defines a Circular ROI, accessible via the :meth:`~CircleMode.roi` method
    """

    icon = 'glue_circle'
    tool_id = 'select:circle'
    action_text = 'Circular ROI'
    tool_tip = 'Define a circular region of interest'
    shortcut = 'C'

    def __init__(self, viewer, **kwargs):
        super(CircleMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtCircularROI(self._axes)


@viewer_tool
class PolyMode(ClickRoiMode):
    """
    Defines a Polygonal ROI, accessible via the :meth:`~PolyMode.roi` method
    """

    icon = 'glue_lasso'
    tool_id = 'select:polygon'
    action_text = 'Polygonal ROI'
    tool_tip = ('Lasso a region of interest\n'
                '  ENTER accepts the path\n'
                '  ESCAPE clears the path')
    status_tip = ('CLICK and DRAG (or CLICK multiple times) to define selection,'
                  ' ENTER to finalize, ESC to cancel, CTRL-CLICK and DRAG to move selection')
    shortcut = 'G'

    def __init__(self, viewer, **kwargs):
        super(PolyMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtPolygonalROI(self._axes)


# TODO: determine why LassoMode exists since it's the same as PolyMode?

@viewer_tool
class LassoMode(RoiMode):
    """
    Defines a Polygonal ROI, accessible via the :meth:`~LassoMode.roi` method
    """

    icon = 'glue_lasso'
    tool_id = 'select:lasso'
    action_text = 'Polygonal ROI'
    tool_tip = ('Lasso a region of interest\n'
                '  ENTER accepts the path\n'
                '  ESCAPE clears the path')
    status_tip = ('CLICK and DRAG (or CLICK multiple times) to define selection,'
                  ' ENTER to finalize, ESC to cancel, CTRL-CLICK and DRAG to move selection')
    shortcut = 'L'

    def __init__(self, viewer, **kwargs):
        super(LassoMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtPolygonalROI(self._axes)


@viewer_tool
class HRangeMode(RoiMode):
    """
    Defines a Range ROI, accessible via the :meth:`~HRangeMode.roi` method.

    This class defines horizontal ranges
    """

    icon = 'glue_xrange_select'
    tool_id = 'select:xrange'
    action_text = 'X range'
    tool_tip = 'Select a range of x values'
    shortcut = 'X'

    def __init__(self, viewer, **kwargs):
        super(HRangeMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtXRangeROI(self._axes)


@viewer_tool
class VRangeMode(RoiMode):
    """
    Defines a Range ROI, accessible via the :meth:`~VRangeMode.roi` method.

    This class defines vertical ranges.
    """

    icon = 'glue_yrange_select'
    tool_id = 'select:yrange'
    action_text = 'Y range'
    tool_tip = 'Select a range of y values'
    shortcut = 'Y'

    def __init__(self, viewer, **kwargs):
        super(VRangeMode, self).__init__(viewer, **kwargs)
        self._roi_tool = qt_roi.QtYRangeROI(self._axes)


@viewer_tool
class PickMode(RoiMode):
    """
    Defines a PointROI.

    Defines single point selections.
    """

    icon = 'glue_point'
    tool_id = 'select:pick'
    action_text = 'Click on the item to select'
    tool_tip = 'Select a single item'
    shortcut = 'K'

    def __init__(self, viewer, **kwargs):
        super(PickMode, self).__init__(viewer, **kwargs)
        self._roi_tool = roi.MplPickROI(self._axes)

    def press(self, event):
        super(PickMode, self).press(event)
        self._drag = True


@viewer_tool
class ContrastMode(ToolbarModeBase):
    """
    Uses right mouse button drags to set bias and contrast, DS9-style.

    The horizontal position of the mouse sets the bias, the vertical position
    sets the contrast.

    The move_callback defaults to calling _set_norm on the viewer with the
    instance of ConstrastMode as the sole argument.
    """

    icon = 'glue_contrast'
    tool_id = 'image:contrast'
    action_text = 'Contrast'
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

        a = QtWidgets.QAction("sqrt", None)
        a.triggered.connect(nonpartial(setattr, self, 'stretch', 'sqrt'))
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


class ColormapAction(QtWidgets.QAction):

    def __init__(self, label, cmap, parent):
        super(ColormapAction, self).__init__(label, parent)
        self.cmap = cmap
        pm = cmap2pixmap(cmap)
        self.setIcon(QtGui.QIcon(pm))


@viewer_tool
class ColormapMode(Tool):
    """
    A tool to change the colormap used in a viewer.

    This calls a ``set_cmap`` method on the viewer, which should take the name
    of the colormap as the sole argument.
    """

    icon = 'glue_rainbow'
    tool_id = 'image:colormap'
    action_text = 'Set color scale'
    tool_tip = 'Set color scale'

    def menu_actions(self):
        from glue import config
        acts = []
        for label, cmap in config.colormaps:
            a = ColormapAction(label, cmap, self.viewer)
            a.triggered.connect(nonpartial(self.viewer.set_cmap, cmap))
            acts.append(a)
        return acts
