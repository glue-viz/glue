"""
Tests for the backend-neutral matplotlib path slicer modes in
:mod:`glue.plugins.tools.path_slicer.matplotlib_mode`.

These run against the Agg-backed :class:`SimpleImageViewer` so the
full PathMode + ToolbarModeBase inheritance chain is exercised. Only
test-internal stubs (e.g. a one-off subclass that sets the required
tool ID / viewer class) are mocked.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from glue.core import Data
from glue.core.application_base import Application
from glue.core.coordinates import IdentityCoordinates
from glue.plugins.tools.path_slicer.matplotlib_mode import (
    BasePathSlicerCrosshairMode, BasePathSlicerMode)
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData
from glue.viewers.image.viewer import SimpleImageViewer


def _make_app_with_cube_viewer(shape=(6, 5, 4)):
    cube = Data(label='cube',
                x=np.arange(np.prod(shape), dtype=float).reshape(shape),
                coords=IdentityCoordinates(n_dim=len(shape)))
    app = Application()
    app.data_collection.append(cube)
    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(cube)
    # MouseMode reads viewer.central_widget.canvas; SimpleImageViewer
    # doesn't expose one because there's no UI layer, so point it at
    # the matplotlib figure (which exposes the canvas).
    viewer.central_widget = viewer.figure
    return app, cube, viewer


def _make_path_slice_and_slice_viewer(app, cube, cube_viewer,
                                      vx=(0., 1., 2., 3.),
                                      vy=(0., 1., 2., 3.)):
    path_slice = PathSlicedData(
        cube,
        cube_viewer.state.x_att, list(vx),
        cube_viewer.state.y_att, list(vy))
    path_slice.parent_viewer = cube_viewer
    app.data_collection.append(path_slice)
    slice_viewer = app.new_data_viewer(SimpleImageViewer)
    slice_viewer.add_data(path_slice)
    slice_viewer.central_widget = slice_viewer.figure
    return path_slice, slice_viewer


def _fake_mode_class(slice_viewer_cls=SimpleImageViewer, tool_id='test:slice'):
    """A throwaway :class:`BasePathSlicerMode` subclass with the
    required class attributes set, for instantiation in tests."""
    class _FakeMode(BasePathSlicerMode):
        pass
    _FakeMode.tool_id = tool_id
    _FakeMode.slice_viewer_cls = slice_viewer_cls
    return _FakeMode


class TestBasePathSlicerMode:

    def test_init_rejects_subclass_without_slice_viewer_cls(self):
        _, _, viewer = _make_app_with_cube_viewer()

        class _Bare(BasePathSlicerMode):
            tool_id = 'test:bare'
            # slice_viewer_cls left at None
        with pytest.raises(TypeError, match='slice_viewer_cls'):
            _Bare(viewer)

    def test_enabled_for_3d_reference_data(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = _fake_mode_class()(viewer)
        assert mode.enabled is True

    def test_disabled_for_non_3d_reference_data(self):
        # Build a 2-D data and viewer.
        image = Data(x=np.arange(20.).reshape((4, 5)),
                     coords=IdentityCoordinates(n_dim=2))
        app = Application()
        app.data_collection.append(image)
        viewer = app.new_data_viewer(SimpleImageViewer)
        viewer.add_data(image)
        viewer.central_widget = viewer.figure
        mode = _fake_mode_class()(viewer)
        assert mode.enabled is False

    def test_reference_data_callback_is_safe_after_close(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = _fake_mode_class()(viewer)
        mode.viewer = None  # Tool.close() does this
        # The state callback fires after close; must not raise.
        mode._on_reference_data_change()

    def test_extract_callback_routes_to_open_or_update(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = _fake_mode_class()(viewer)
        mode._open_or_update = MagicMock()
        roi = MagicMock()
        roi.to_polygon.return_value = ([1, 2, 3], [0, 1, 2])
        roi_mode = MagicMock()
        roi_mode.roi.return_value = roi
        mode._extract_callback(roi_mode)
        mode._open_or_update.assert_called_once_with([1, 2, 3], [0, 1, 2])

    def test_close_closes_slice_viewer(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = _fake_mode_class()(viewer)
        fake_slice_viewer = MagicMock()
        mode._slice_viewer = fake_slice_viewer
        mode.close()
        fake_slice_viewer.close.assert_called_once()
        assert mode._slice_viewer is None

    def test_close_is_safe_without_slice_viewer(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = _fake_mode_class()(viewer)
        assert mode._slice_viewer is None
        mode.close()  # should not raise

    def test_open_or_update_opens_real_slice_viewer(self):
        # Exercise the full open_or_update_slice_viewer round-trip --
        # SimpleImageViewer is happy to be the slice viewer class.
        app, cube, viewer = _make_app_with_cube_viewer()
        mode = _fake_mode_class(slice_viewer_cls=SimpleImageViewer)(viewer)
        mode._open_or_update([1., 2., 3.], [0., 1., 2.])
        assert isinstance(mode._slice_viewer, SimpleImageViewer)
        # A PathSlicedData was created and added as a layer.
        layers = [ls.layer for ls in mode._slice_viewer.state.layers]
        assert any(isinstance(d, PathSlicedData) for d in layers)


# ---------------------------------------------------------------------------
# BasePathSlicerCrosshairMode
# ---------------------------------------------------------------------------


class TestBasePathSlicerCrosshairMode:

    def test_disabled_for_non_path_sliced_reference_data(self):
        _, _, viewer = _make_app_with_cube_viewer()
        # The cube viewer's reference_data is a plain Data, not a PV.
        mode = BasePathSlicerCrosshairMode(viewer)
        assert mode.enabled is False
        assert mode.data is None

    def test_disabled_for_path_slice_without_parent_viewer(self):
        app, cube, cube_viewer = _make_app_with_cube_viewer()
        path_slice = PathSlicedData(
            cube,
            cube_viewer.state.x_att, [0., 1., 2.],
            cube_viewer.state.y_att, [0., 1., 2.])
        # Intentionally do NOT set path_slice.parent_viewer.
        app.data_collection.append(path_slice)
        slice_viewer = app.new_data_viewer(SimpleImageViewer)
        slice_viewer.add_data(path_slice)
        slice_viewer.central_widget = slice_viewer.figure
        mode = BasePathSlicerCrosshairMode(slice_viewer)
        assert mode.enabled is False

    def test_enabled_for_path_slice_with_parent_viewer(self):
        app, cube, cube_viewer = _make_app_with_cube_viewer()
        path_slice, slice_viewer = _make_path_slice_and_slice_viewer(
            app, cube, cube_viewer)
        mode = BasePathSlicerCrosshairMode(slice_viewer)
        assert mode.enabled is True
        assert mode.data is path_slice

    def test_reference_data_callback_safe_after_close(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = BasePathSlicerCrosshairMode(viewer)
        mode.viewer = None
        mode._on_reference_data_change()  # must not raise

    def test_press_and_release_toggle_active(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = BasePathSlicerCrosshairMode(viewer)
        assert mode._active is False
        mode._on_press(MagicMock())
        assert mode._active is True
        mode._on_release(MagicMock())
        assert mode._active is False

    def test_move_does_nothing_when_inactive(self):
        app, cube, cube_viewer = _make_app_with_cube_viewer()
        _, slice_viewer = _make_path_slice_and_slice_viewer(
            app, cube, cube_viewer)
        mode = BasePathSlicerCrosshairMode(slice_viewer)
        slices_before = tuple(cube_viewer.state.slices)
        mode._event_xdata = 1.0
        mode._event_ydata = 2.0
        mode._on_move(MagicMock())
        # _active is still False; nothing should have moved.
        assert tuple(cube_viewer.state.slices) == slices_before

    def test_move_does_nothing_when_event_coords_missing(self):
        app, cube, cube_viewer = _make_app_with_cube_viewer()
        _, slice_viewer = _make_path_slice_and_slice_viewer(
            app, cube, cube_viewer)
        mode = BasePathSlicerCrosshairMode(slice_viewer)
        mode._active = True
        mode._event_xdata = None
        mode._event_ydata = None
        slices_before = tuple(cube_viewer.state.slices)
        mode._on_move(MagicMock())
        assert tuple(cube_viewer.state.slices) == slices_before

    def test_move_updates_crosshair_and_parent_slice(self):
        app, cube, cube_viewer = _make_app_with_cube_viewer()
        _, slice_viewer = _make_path_slice_and_slice_viewer(
            app, cube, cube_viewer)
        mode = BasePathSlicerCrosshairMode(slice_viewer)
        mode.activate()
        mode._active = True
        mode._event_xdata = 2.0
        mode._event_ydata = 3.0
        mode._on_move(MagicMock())
        # The slice axis on the parent cube viewer should have moved.
        slice_axis = next(
            i for i in range(cube.ndim)
            if i not in (cube_viewer.state.x_att.axis,
                         cube_viewer.state.y_att.axis))
        assert cube_viewer.state.slices[slice_axis] == 3
        # The crosshair artist tracks the cursor projection.
        assert len(mode._crosshair.get_xdata()) == 1
        mode.deactivate()

    def test_activate_then_deactivate_round_trip(self):
        app, cube, cube_viewer = _make_app_with_cube_viewer()
        path_slice, slice_viewer = _make_path_slice_and_slice_viewer(
            app, cube, cube_viewer)
        mode = BasePathSlicerCrosshairMode(slice_viewer)
        n_lines_before = len(cube_viewer.axes.get_lines())
        mode.activate()
        # activate adds the path line + the crosshair marker line to the
        # parent viewer's axes.
        assert len(cube_viewer.axes.get_lines()) > n_lines_before
        assert mode._line is not None
        assert mode._crosshair is not None
        mode.deactivate()
        # deactivate removes them again.
        assert len(cube_viewer.axes.get_lines()) == n_lines_before
        assert mode._line is None
        assert mode._crosshair is None

    def test_deactivate_is_safe_when_never_activated(self):
        _, _, viewer = _make_app_with_cube_viewer()
        mode = BasePathSlicerCrosshairMode(viewer)
        # _line / _crosshair never assigned; deactivate must not raise.
        mode.deactivate()
