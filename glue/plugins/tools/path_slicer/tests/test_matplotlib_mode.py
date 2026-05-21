"""
Tests for the backend-neutral matplotlib path slicer modes in
:mod:`glue.plugins.tools.path_slicer.matplotlib_mode`.

The modes inherit matplotlib UI behaviour from :class:`PathMode` /
:class:`ToolbarModeBase` and operate against a real image viewer in
glue-qt and glue-jupyter. Here we test the path-slicer-specific
additions only, with the viewer stubbed out via :class:`MagicMock`.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from glue.core import Data, DataCollection
from glue.core.coordinates import IdentityCoordinates
from glue.plugins.tools.path_slicer.matplotlib_mode import (
    BasePathSlicerCrosshairMode, BasePathSlicerMode)
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData


def _make_cube_viewer():
    """Return a MagicMock dressed up enough to satisfy PathMode's
    __init__ and the BasePathSlicerMode callback setup."""
    viewer = MagicMock()
    viewer.axes.figure.canvas.get_width_height.return_value = (640, 480)
    viewer.state.reference_data = MagicMock(ndim=3)
    return viewer


def _make_fake_subclass(viewer_cls=None):
    """Return a fresh BasePathSlicerMode subclass with tool_id /
    slice_viewer_cls set, so tests can instantiate it."""
    class _FakeMode(BasePathSlicerMode):
        tool_id = 'test:slice'
        slice_viewer_cls = viewer_cls if viewer_cls is not None else MagicMock()
    return _FakeMode


class TestBasePathSlicerMode:

    def test_init_rejects_subclass_without_slice_viewer_cls(self):
        class _Bare(BasePathSlicerMode):
            tool_id = 'test:bare'
            # slice_viewer_cls left at None
        with pytest.raises(TypeError, match='slice_viewer_cls'):
            _Bare(_make_cube_viewer())

    def test_enabled_for_3d_reference_data(self):
        viewer = _make_cube_viewer()
        viewer.state.reference_data.ndim = 3
        mode = _make_fake_subclass()(viewer)
        assert mode.enabled is True

    def test_disabled_for_non_3d_reference_data(self):
        viewer = _make_cube_viewer()
        viewer.state.reference_data.ndim = 2
        mode = _make_fake_subclass()(viewer)
        assert mode.enabled is False

    def test_reference_data_callback_is_safe_after_close(self):
        viewer = _make_cube_viewer()
        mode = _make_fake_subclass()(viewer)
        mode.viewer = None  # Tool.close() does this
        # The state callback fires after close; must not raise.
        mode._on_reference_data_change()

    def test_extract_callback_routes_to_open_or_update(self):
        # Stub out the open_or_update helper that _extract_callback delegates
        # to and verify it gets the vx, vy from the ROI's polygon.
        viewer = _make_cube_viewer()
        mode = _make_fake_subclass()(viewer)
        mode._open_or_update = MagicMock()

        roi = MagicMock()
        roi.to_polygon.return_value = ([1, 2, 3], [0, 1, 2])
        roi_mode = MagicMock()
        roi_mode.roi.return_value = roi
        mode._extract_callback(roi_mode)

        mode._open_or_update.assert_called_once_with([1, 2, 3], [0, 1, 2])

    def test_close_closes_slice_viewer(self):
        viewer = _make_cube_viewer()
        mode = _make_fake_subclass()(viewer)
        fake_slice_viewer = MagicMock()
        mode._slice_viewer = fake_slice_viewer
        mode.close()
        fake_slice_viewer.close.assert_called_once()
        assert mode._slice_viewer is None

    def test_close_is_safe_without_slice_viewer(self):
        viewer = _make_cube_viewer()
        mode = _make_fake_subclass()(viewer)
        assert mode._slice_viewer is None
        mode.close()  # should not raise

    def test_open_or_update_delegates_to_helper(self, monkeypatch):
        from glue.plugins.tools.path_slicer import matplotlib_mode
        viewer = _make_cube_viewer()
        mode = _make_fake_subclass()(viewer)
        fake_helper = MagicMock(return_value='returned-slice-viewer')
        monkeypatch.setattr(matplotlib_mode, 'open_or_update_slice_viewer',
                            fake_helper)
        mode._open_or_update([1, 2, 3], [0, 1, 2])
        fake_helper.assert_called_once_with(
            viewer, None, mode.slice_viewer_cls, [1, 2, 3], [0, 1, 2])
        assert mode._slice_viewer == 'returned-slice-viewer'


# ---------------------------------------------------------------------------
# BasePathSlicerCrosshairMode
# ---------------------------------------------------------------------------


def _make_path_slice(parent_viewer=None):
    cube = Data(label='cube',
                x=np.arange(120., dtype=float).reshape((6, 5, 4)),
                coords=IdentityCoordinates(n_dim=3))
    dc = DataCollection([cube])
    path_slice = PathSlicedData(
        cube,
        cube.pixel_component_ids[1], [0., 1., 2., 3.],
        cube.pixel_component_ids[2], [0., 1., 2., 3.])
    dc.append(path_slice)
    if parent_viewer is not None:
        path_slice.parent_viewer = parent_viewer
        parent_viewer.state.reference_data = cube
        parent_viewer.state.x_att = cube.pixel_component_ids[1]
        parent_viewer.state.y_att = cube.pixel_component_ids[2]
        parent_viewer.state.slices = (0, 0, 0)
    return path_slice


class TestBasePathSlicerCrosshairMode:

    def test_disabled_for_non_path_sliced_reference_data(self):
        viewer = MagicMock()
        viewer.state.reference_data = MagicMock()  # not a PathSlicedData
        mode = BasePathSlicerCrosshairMode(viewer)
        assert mode.enabled is False
        assert mode.data is None

    def test_disabled_for_path_slice_without_parent_viewer(self):
        viewer = MagicMock()
        viewer.state.reference_data = _make_path_slice()
        mode = BasePathSlicerCrosshairMode(viewer)
        assert mode.enabled is False

    def test_enabled_for_path_slice_with_parent_viewer(self):
        viewer = MagicMock()
        parent = MagicMock()
        path_slice = _make_path_slice(parent_viewer=parent)
        viewer.state.reference_data = path_slice
        mode = BasePathSlicerCrosshairMode(viewer)
        assert mode.enabled is True
        assert mode.data is path_slice

    def test_reference_data_callback_safe_after_close(self):
        viewer = MagicMock()
        viewer.state.reference_data = MagicMock()
        mode = BasePathSlicerCrosshairMode(viewer)
        mode.viewer = None
        mode._on_reference_data_change()  # must not raise

    def test_press_and_release_toggle_active(self):
        viewer = MagicMock()
        viewer.state.reference_data = MagicMock()
        mode = BasePathSlicerCrosshairMode(viewer)
        assert mode._active is False
        mode._on_press(MagicMock())
        assert mode._active is True
        mode._on_release(MagicMock())
        assert mode._active is False

    def test_move_does_nothing_when_inactive(self):
        viewer = MagicMock()
        parent = MagicMock()
        viewer.state.reference_data = _make_path_slice(parent_viewer=parent)
        mode = BasePathSlicerCrosshairMode(viewer)
        # active is False by default; move must be a no-op (no exception,
        # no parent draw call).
        parent.figure.canvas.draw_idle.reset_mock()
        mode._on_move(MagicMock())
        parent.figure.canvas.draw_idle.assert_not_called()

    def test_move_does_nothing_when_event_coords_missing(self):
        # _event_xdata/_event_ydata may be None when the cursor leaves
        # the axes; the move callback must bail.
        viewer = MagicMock()
        parent = MagicMock()
        viewer.state.reference_data = _make_path_slice(parent_viewer=parent)
        mode = BasePathSlicerCrosshairMode(viewer)
        mode._active = True
        mode._event_xdata = None
        mode._event_ydata = None
        parent.figure.canvas.draw_idle.reset_mock()
        mode._on_move(MagicMock())
        parent.figure.canvas.draw_idle.assert_not_called()

    def test_move_updates_crosshair_and_parent_slice(self):
        viewer = MagicMock()
        parent = MagicMock()
        path_slice = _make_path_slice(parent_viewer=parent)
        viewer.state.reference_data = path_slice
        mode = BasePathSlicerCrosshairMode(viewer)
        mode._active = True
        mode._crosshair = MagicMock()
        mode._event_xdata = 2.0
        mode._event_ydata = 3.0
        mode._on_move(MagicMock())
        mode._crosshair.set_xdata.assert_called_once()
        mode._crosshair.set_ydata.assert_called_once()
        # drive_parent_slice should have written to state.slices --
        # axis 0 is non-displayed here so it gets the int(3.0).
        assert parent.state.slices[0] == 3
        parent.figure.canvas.draw_idle.assert_called_once()

    def test_deactivate_is_safe_when_never_activated(self):
        viewer = MagicMock()
        viewer.state.reference_data = MagicMock()  # not a PathSlicedData
        mode = BasePathSlicerCrosshairMode(viewer)
        # _line / _crosshair never assigned; deactivate must not raise.
        mode.deactivate()

    def test_activate_then_deactivate_round_trip(self):
        viewer = MagicMock()
        parent = MagicMock()
        path_slice = _make_path_slice(parent_viewer=parent)
        viewer.state.reference_data = path_slice
        mode = BasePathSlicerCrosshairMode(viewer)
        mode.activate()
        # The line is registered on the parent axes and a Line2D was
        # returned by axes.plot() for the crosshair.
        parent.axes.add_line.assert_called_once()
        parent.axes.plot.assert_called_once()
        assert mode._line is not None
        assert mode._crosshair is not None

        mode._line.remove = MagicMock()
        mode._crosshair.remove = MagicMock()
        mode.deactivate()
        # The artist .remove() methods are called and references are
        # cleared.
        assert mode._line is None
        assert mode._crosshair is None
