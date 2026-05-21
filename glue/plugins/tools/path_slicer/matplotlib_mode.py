"""
Backend-neutral matplotlib path slicer modes.

Both the Qt and Jupyter (matplotlib) image viewers expose a path-slicer
tool that lets the user draw a path on a cube and have the values along
that path materialised as a :class:`PathSlicedData` shown in a fresh slice
image viewer. The two front-ends differ only in which viewer class
``new_data_viewer`` is called with; everything else is identical
matplotlib code. This module provides two abstract base classes -- one
for the path-drawing tool, one for the crosshair tool on the slice viewer
-- which each front-end subclasses to set its viewer class and tool ID
and register via ``@viewer_tool``.
"""
import numpy as np
from matplotlib.lines import Line2D

from glue.viewers.matplotlib.toolbar_mode import PathMode, ToolbarModeBase

from .common import drive_parent_slice
from .multi_trace import MultiTracePathSlicerMixin
from .path_sliced_data import PathSlicedData


__all__ = ['BasePathSlicerMode', 'BasePathSlicerCrosshairMode']


_PATH_COLOR = '#669dff'
_PATH_ALPHA_ACTIVE = 1.0
_PATH_ALPHA_INACTIVE = 0.3


class BasePathSlicerMode(MultiTracePathSlicerMixin, PathMode):
    """
    Matplotlib-backed path slicer tool. Multi-trace bookkeeping
    (``self._traces``, ``self._target_trace``, ``self.menu_entries()``,
    ``self.set_target()``) comes from
    :class:`MultiTracePathSlicerMixin`; this class adds the path-
    drawing inheritance from :class:`PathMode` and renders the per-
    trace path overlay on the source viewer's matplotlib axes.

    Subclasses set :attr:`slice_viewer_cls` and :attr:`tool_id`,
    decorate themselves with ``@viewer_tool``, and may override
    :meth:`_on_traces_changed` to refresh their backend's UI.
    """

    icon = 'glue_slice'
    action_text = 'Slice Extraction'
    tool_tip = ('Extract a slice from an arbitrary path\n'
                '  ENTER accepts the path\n'
                '  ESCAPE clears the path')
    status_tip = ('Draw a path then press ENTER to extract slice, '
                  'or press ESC to cancel')
    shortcut = 'P'

    def __init__(self, viewer, **kwargs):
        if self.slice_viewer_cls is None:
            raise TypeError(
                f"{type(self).__name__} must set 'slice_viewer_cls' "
                "(the viewer class to open for the path slice)")
        super().__init__(viewer, **kwargs)
        self._init_multi_trace()
        self._roi_callback = self._extract_callback
        # Overlay artists on the source viewer's axes, keyed by trace
        # identity.
        self._overlays = {}
        self.viewer.state.add_callback('reference_data',
                                       self._on_reference_data_change)
        self._on_reference_data_change()

    def _on_reference_data_change(self, *args):
        # State callbacks can fire after Tool.close() clears self.viewer.
        if self.viewer is None:
            return
        if self.viewer.state.reference_data is not None:
            self.enabled = self.viewer.state.reference_data.ndim == 3

    def _extract_callback(self, mode):
        vx, vy = mode.roi().to_polygon()
        self._open_or_update(vx, vy)

    # ------------------------------------------------------------------
    # Matplotlib overlay drawing (mixin hooks)
    # ------------------------------------------------------------------

    def _refresh_overlays(self):
        # Remove any artists for traces that no longer exist.
        current_keys = {id(trace) for trace in self._traces}
        for key in list(self._overlays):
            if key not in current_keys:
                self._overlays.pop(key).remove()

        for trace in self._traces:
            key = id(trace)
            x, y = trace[0].x, trace[0].y
            alpha = (_PATH_ALPHA_ACTIVE if trace is self._target_trace
                     else _PATH_ALPHA_INACTIVE)
            if key in self._overlays:
                line = self._overlays[key]
                line.set_data(x, y)
                line.set_alpha(alpha)
            else:
                line = Line2D(x, y, color=_PATH_COLOR, alpha=alpha,
                              lw=2, zorder=100)
                self.viewer.axes.add_line(line)
                self._overlays[key] = line
        self.viewer.figure.canvas.draw_idle()

    def hover_preview(self, target):
        for trace in self._traces:
            line = self._overlays.get(id(trace))
            if line is None:
                continue
            line.set_alpha(_PATH_ALPHA_ACTIVE if trace is target
                           else _PATH_ALPHA_INACTIVE)
        self.viewer.figure.canvas.draw_idle()

    def close(self):
        for line in self._overlays.values():
            line.remove()
        self._overlays.clear()
        self._close_slice_viewers()
        return super().close()


class BasePathSlicerCrosshairMode(ToolbarModeBase):
    """
    Base class for the crosshair tool on the slice viewer. While the
    mouse is dragged, draws the path on the parent cube viewer,
    highlights the cursor's projection back to parent pixel
    coordinates, and pushes the cursor's slice y onto the parent viewer's
    ``state.slices``.

    Subclasses set :attr:`tool_id` and decorate themselves with
    ``@viewer_tool``.
    """

    icon = 'glue_path'
    action_text = 'Show position on original path'
    tool_tip = 'Click and drag to show position of cursor on original slice.'
    status_tip = tool_tip

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._move_callback = self._on_move
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._active = False
        self._line = None
        self._crosshair = None
        self.data = None
        self.viewer.state.add_callback('reference_data',
                                       self._on_reference_data_change)
        self._on_reference_data_change()

    def _on_reference_data_change(self, *args):
        if self.viewer is None:
            return
        ref = self.viewer.state.reference_data
        self.enabled = isinstance(ref, PathSlicedData) \
            and getattr(ref, 'parent_viewer', None) is not None
        self.data = ref if self.enabled else None

    def activate(self):
        self._line = Line2D(self.data.x, self.data.y, zorder=1000,
                            color='#669dff', alpha=0.6, lw=2)
        self.data.parent_viewer.axes.add_line(self._line)
        self._crosshair = self.data.parent_viewer.axes.plot(
            [], [], '+', ms=12, mfc='none', mec='#669dff', mew=1,
            zorder=100)[0]
        self.data.parent_viewer.figure.canvas.draw_idle()
        super().activate()

    def deactivate(self):
        if self._line is not None:
            self._line.remove()
            self._line = None
        if self._crosshair is not None:
            self._crosshair.remove()
            self._crosshair = None
        if self.data is not None:
            self.data.parent_viewer.figure.canvas.draw_idle()
        super().deactivate()

    def _on_press(self, mode):
        self._active = True

    def _on_release(self, mode):
        self._active = False

    def _on_move(self, mode):
        if not self._active or self.data is None:
            return
        xdata, ydata = self._event_xdata, self._event_ydata
        if xdata is None or ydata is None:
            return
        ind = round(np.clip(xdata, 0, self.data.shape[-1] - 1))
        x = self.data.x[ind]
        y = self.data.y[ind]
        self._crosshair.set_xdata([x])
        self._crosshair.set_ydata([y])
        drive_parent_slice(self.data, ydata)
        self.data.parent_viewer.figure.canvas.draw_idle()
