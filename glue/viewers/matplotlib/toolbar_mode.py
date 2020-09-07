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

All the mouse modes and tools in this file are Matplotlib-specific.
"""

from glue.core import roi
from glue.viewers.common.tool import CheckableTool
from glue.config import viewer_tool
from .mouse_mode import MouseMode


__all__ = ['ToolbarModeBase', 'RoiModeBase', 'RoiMode', 'PersistentRoiMode',
           'ClickRoiMode', 'RectangleMode', 'PathMode', 'CircleMode',
           'PolyMode', 'HRangeMode', 'VRangeMode', 'PickMode']


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
    disable_on_finalize = True

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
        # For persistent ROIs, the user might e.g. pan and zoom around before
        # the selection is finalized. The Matplotlib ROIs cache the image
        # background to make things more efficient, but if the user pans/zooms
        # we need to make sure we reset the background.
        if getattr(self._roi_tool, '_mid_selection', False):
            self._roi_tool._reset_background()
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
        if self.disable_on_finalize:
            self.viewer.toolbar.active_tool = None

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

    def _update_drag(self, event):
        if self._drag or self._start_event is None:
            return

        dx = abs(event.x - self._start_event.x)
        dy = abs(event.y - self._start_event.y)

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
        data_space = not hasattr(viewer.state, 'plot_mode') or viewer.state.plot_mode == 'rectilinear'
        self._roi_tool = roi.MplRectangularROI(self._axes, data_space=data_space)


class PathMode(ClickRoiMode):

    persistent = True

    def __init__(self, viewer, **kwargs):
        super(PathMode, self).__init__(viewer, **kwargs)
        self._roi_tool = roi.MplPathROI(self._axes)

        self._roi_tool.plot_opts.update(color='#de2d26',
                                        fill=False,
                                        linewidth=3,
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
        data_space = not hasattr(viewer.state, 'plot_mode') or viewer.state.plot_mode == 'rectilinear'
        self._roi_tool = roi.MplCircularROI(self._axes, data_space=data_space)


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
        data_space = not hasattr(viewer.state, 'plot_mode') or viewer.state.plot_mode == 'rectilinear'
        self._roi_tool = roi.MplPolygonalROI(self._axes, data_space=data_space)


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
        data_space = not hasattr(viewer.state, 'plot_mode') or viewer.state.plot_mode == 'rectilinear'
        self._roi_tool = roi.MplXRangeROI(self._axes, data_space=data_space)


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
        data_space = not hasattr(viewer.state, 'plot_mode') or viewer.state.plot_mode == 'rectilinear'
        self._roi_tool = roi.MplYRangeROI(self._axes, data_space=data_space)


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
