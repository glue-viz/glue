"""
Tests for the backend-agnostic helpers in
:mod:`glue.plugins.tools.path_slicer.common`.

Where the helpers need a viewer we use :class:`SimpleImageViewer` (the
Agg-backed image viewer in glue-core) so the data-side and viewer-side
plumbing both run through real glue code. The pure data helpers don't
need a viewer at all. End-to-end coverage against the Qt and Jupyter
image viewers lives in glue-qt and glue-jupyter.
"""

import numpy as np
import pytest

from glue.core import Data
from glue.core.application_base import Application
from glue.core.coordinates import IdentityCoordinates
from glue.plugins.tools.path_slicer.common import drive_parent_slice
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData
from glue.viewers.image.viewer import SimpleImageViewer


def _make_cube(label='cube', shape=(6, 5, 4)):
    return Data(label=label,
                x=np.arange(np.prod(shape), dtype=float).reshape(shape),
                coords=IdentityCoordinates(n_dim=len(shape)))


def _make_app_with_cube_viewer(cube=None):
    """Stand up an :class:`Application` with a single cube loaded into a
    :class:`SimpleImageViewer`."""
    cube = cube or _make_cube()
    app = Application()
    app.data_collection.append(cube)
    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(cube)
    return app, cube, viewer


class TestDriveParentSlice:
    """``drive_parent_slice`` (uses SimpleImageViewer)."""

    def test_writes_to_non_displayed_axis(self):
        _app, cube, viewer = _make_app_with_cube_viewer()
        # The SimpleImageViewer defaults pick the two highest axes as
        # x/y, so axis 0 is the non-displayed (slice) axis here.
        path_slice = PathSlicedData(
            cube,
            viewer.state.x_att, [0., 1., 2.],
            viewer.state.y_att, [0., 1., 2.])
        path_slice.parent_viewer = viewer
        before = tuple(viewer.state.slices)
        drive_parent_slice(path_slice, 4.0)
        after = tuple(viewer.state.slices)
        # The slice axis (the one that isn't x_att or y_att) is set to int(4).
        slice_axis = next(i for i in range(cube.ndim)
                          if i not in (viewer.state.x_att.axis,
                                       viewer.state.y_att.axis))
        assert after[slice_axis] == 4
        assert after != before

    def test_truncates_float_y_to_int(self):
        _app, cube, viewer = _make_app_with_cube_viewer()
        path_slice = PathSlicedData(
            cube,
            viewer.state.x_att, [0., 1., 2.],
            viewer.state.y_att, [0., 1., 2.])
        path_slice.parent_viewer = viewer
        drive_parent_slice(path_slice, 2.7)
        slice_axis = next(i for i in range(cube.ndim)
                          if i not in (viewer.state.x_att.axis,
                                       viewer.state.y_att.axis))
        # int() truncates toward zero.
        assert viewer.state.slices[slice_axis] == 2


class TestPathRelativeLinkValidation:
    """``PathRelativeLink`` validation errors not reachable through the happy path."""

    def test_init_rejects_non_path_sliced_data(self):
        from glue.plugins.tools.path_slicer.path_sliced_data_links import (
            PathRelativeLink)
        cube = _make_cube()
        with pytest.raises(TypeError, match='PathSlicedData'):
            PathRelativeLink(cube, cube)


class TestLinkGroupValidation:
    """``link_path_sliced_group`` validation errors."""

    def test_link_group_rejects_single_input(self):
        from glue.plugins.tools.path_slicer.path_sliced_data_links import (
            link_path_sliced_group)
        app = Application()
        cube = _make_cube()
        app.data_collection.append(cube)
        path_slice = PathSlicedData(
            cube,
            cube.pixel_component_ids[1], [0., 1., 2.],
            cube.pixel_component_ids[2], [0., 1., 2.])
        app.data_collection.append(path_slice)
        with pytest.raises(ValueError, match='at least two'):
            link_path_sliced_group(app.data_collection, path_slice)
