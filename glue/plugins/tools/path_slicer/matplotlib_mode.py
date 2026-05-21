"""
Backend-neutral matplotlib path slicer modes.

Both the Qt and Jupyter (matplotlib) image viewers expose a path-slicer
tool that lets the user draw a path on a cube and have the values along
that path materialised as a :class:`PathSlicedData` shown in a fresh PV
image viewer. The two front-ends differ only in which viewer class
``new_data_viewer`` is called with; everything else is identical
matplotlib code. This module provides two abstract base classes -- one
for the path-drawing tool, one for the crosshair tool on the PV viewer
-- which each front-end subclasses to set its viewer class and tool ID
and register via ``@viewer_tool``.
"""
import numpy as np
from matplotlib.lines import Line2D

from glue.viewers.matplotlib.toolbar_mode import PathMode, ToolbarModeBase

from .common import drive_parent_slice, open_or_update_pv_viewer
from .path_sliced_data import PathSlicedData


__all__ = ['BasePathSlicerMode', 'BasePathSlicerCrosshairMode']


class BasePathSlicerMode(PathMode):
    """
    Base class for the path-drawing tool on a cube viewer. Subclasses
    must set :attr:`pv_viewer_cls` (the viewer class to open for the PV
    slice) and :attr:`tool_id`, and decorate themselves with
    ``@viewer_tool``.
    """

    pv_viewer_cls = None  # set by subclasses

    icon = 'glue_slice'
    action_text = 'Slice Extraction'
    tool_tip = ('Extract a slice from an arbitrary path\n'
                '  ENTER accepts the path\n'
                '  ESCAPE clears the path')
    status_tip = ('Draw a path then press ENTER to extract slice, '
                  'or press ESC to cancel')
    shortcut = 'P'

    def __init__(self, viewer, **kwargs):
        if self.pv_viewer_cls is None:
            raise TypeError(
                f"{type(self).__name__} must set 'pv_viewer_cls' "
                "(the viewer class to open for the PV slice)")
        super().__init__(viewer, **kwargs)
        self._roi_callback = self._extract_callback
        self._pv_viewer = None
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
        self._pv_viewer = open_or_update_pv_viewer(
            self.viewer, self._pv_viewer, self.pv_viewer_cls, vx, vy)

    def close(self):
        if self._pv_viewer is not None:
            self._pv_viewer.close()
            self._pv_viewer = None
        return super().close()


class BasePathSlicerCrosshairMode(ToolbarModeBase):
    """
    Base class for the crosshair tool on the PV viewer. While the
    mouse is dragged, draws the path on the parent cube viewer,
    highlights the cursor's projection back to parent pixel
    coordinates, and pushes the cursor's PV-y onto the parent viewer's
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
