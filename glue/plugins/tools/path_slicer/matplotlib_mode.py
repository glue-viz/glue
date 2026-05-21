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

from .common import (create_trace, drive_parent_slice,
                     open_slice_viewer_for, update_trace)
from .path_sliced_data import PathSlicedData


__all__ = ['BasePathSlicerMode', 'BasePathSlicerCrosshairMode']


_PATH_COLOR = '#669dff'
_PATH_ALPHA_ACTIVE = 1.0
_PATH_ALPHA_INACTIVE = 0.3


class BasePathSlicerMode(PathMode):
    """
    Base class for the path-drawing tool on a cube viewer. Each Enter
    on the tool produces a "trace" -- one :class:`PathSlicedData` per
    Data layer in the source viewer; the trace's PVs are added to a
    fresh slice viewer of :attr:`slice_viewer_cls`. The tool tracks
    every trace it produced on this source viewer (``self._traces``)
    and remembers which trace the next Enter should refresh
    (``self._target_trace``); set ``_target_trace`` to ``None`` to
    create a new trace instead.

    On the matplotlib axes of the source viewer the tool draws one
    overlay line per trace, with the active trace at full opacity and
    the others faded; :meth:`hover_preview` flips the alphas without
    committing the selection so a UI dropdown can preview which path
    is about to be replaced.

    Subclasses set :attr:`slice_viewer_cls` and :attr:`tool_id`,
    decorate themselves with ``@viewer_tool``, and may override
    :meth:`_on_traces_changed` to refresh their backend's UI (e.g. a
    QMenu or ipywidgets.Dropdown).
    """

    slice_viewer_cls = None  # set by subclasses

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
        self._roi_callback = self._extract_callback
        # Parallel lists. ``self._slice_viewer`` shadows the most recent
        # entry of ``self._slice_viewers`` so the crosshair tool's
        # introspection (and existing callers) still find something.
        self._traces = []  # list[list[PathSlicedData]]
        self._slice_viewers = []
        self._slice_viewer = None
        # ``None`` means "create new trace on next Enter"; otherwise one
        # of ``self._traces``.
        self._target_trace = None
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

    def _open_or_update(self, vx, vy):
        if self._target_trace is None:
            new_paths = create_trace(self.viewer, vx, vy, self._traces)
            self._traces.append(new_paths)
            slice_viewer = open_slice_viewer_for(
                self.viewer, self.slice_viewer_cls, new_paths)
            self._slice_viewers.append(slice_viewer)
            self._slice_viewer = slice_viewer
            # The just-created trace becomes the target for the next
            # Enter, so consecutive Enters tweak the same path until
            # the user picks something else from the UI dropdown.
            self._target_trace = self._traces[-1]
        else:
            update_trace(self._target_trace, vx, vy)
        self._refresh_overlays()
        self._on_traces_changed()

    # ------------------------------------------------------------------
    # Public API used by UI dropdowns
    # ------------------------------------------------------------------

    def menu_entries(self):
        """The (label, target) pairs a dropdown UI should show. The
        ``target`` is ``None`` for "Create new path" or a trace from
        :attr:`_traces` for "Update path N"."""
        entries = [('Create new path', None)]
        for i, _trace in enumerate(self._traces, start=1):
            entries.append((f'Update path {i}', _trace))
        return entries

    def set_target(self, target):
        """Set the next-Enter target. ``None`` creates a new trace;
        otherwise must be one of :attr:`_traces`."""
        self._target_trace = target
        self._refresh_overlays()

    def hover_preview(self, target):
        """Temporarily highlight ``target`` (or none if ``target`` is
        ``None``) without committing the selection. UI dropdowns call
        this on hover; :meth:`_refresh_overlays` restores the committed
        state when the menu closes."""
        for trace in self._traces:
            line = self._overlays.get(id(trace))
            if line is None:
                continue
            line.set_alpha(_PATH_ALPHA_ACTIVE if trace is target
                           else _PATH_ALPHA_INACTIVE)
        self.viewer.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Matplotlib overlay drawing
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

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _on_traces_changed(self):
        """Override to refresh the backend's dropdown UI after a trace
        is added or updated. Default no-op."""

    def close(self):
        for line in self._overlays.values():
            line.remove()
        self._overlays.clear()
        for slice_viewer in self._slice_viewers:
            slice_viewer.close()
        self._slice_viewers.clear()
        self._slice_viewer = None
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
