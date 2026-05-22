"""
Tests for the backend-agnostic helpers in
:mod:`glue.plugins.tools.path_slicer.common`.

Where the helpers need a viewer we use :class:`SimpleImageViewer` (the
Agg-backed image viewer in glue-core) so the data-side and viewer-side
plumbing both run through real glue code. The pure data helpers don't
need a viewer at all. End-to-end coverage against the Qt and Jupyter
image viewers lives in glue-qt and glue-jupyter.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from glue.core import Data
from glue.core.application_base import Application
from glue.core.coordinates import IdentityCoordinates
from glue.plugins.tools.path_slicer.common import (
    build_or_update_path_slices, drive_parent_slice, find_existing_path_slice,
    open_or_update_slice_viewer, path_link_exists)
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData
from glue.plugins.tools.path_slicer.path_sliced_data_links import (
    link_path_sliced_pair_paths, link_path_sliced_to_parent)
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


class TestFindExistingPathSlice:
    """``find_existing_path_slice`` (no viewer needed)."""

    def setup_method(self, method):
        self.app = Application()
        self.dc = self.app.data_collection
        self.cube = _make_cube()
        self.dc.append(self.cube)

    def test_returns_none_when_no_slice(self):
        assert find_existing_path_slice(self.dc, self.cube) is None

    def test_returns_match(self):
        path_slice = PathSlicedData(
            self.cube,
            self.cube.pixel_component_ids[1], [0., 1., 2.],
            self.cube.pixel_component_ids[2], [0., 1., 2.])
        self.dc.append(path_slice)
        assert find_existing_path_slice(self.dc, self.cube) is path_slice

    def test_skips_slice_of_other_cube(self):
        other = _make_cube(label='other')
        self.dc.append(other)
        path_slice = PathSlicedData(
            other,
            other.pixel_component_ids[1], [0., 1., 2.],
            other.pixel_component_ids[2], [0., 1., 2.])
        self.dc.append(path_slice)
        assert find_existing_path_slice(self.dc, self.cube) is None


class TestPathLinkExists:
    """``path_link_exists`` (no viewer needed)."""

    def setup_method(self, method):
        self.app = Application()
        self.dc = self.app.data_collection
        self.cube = _make_cube()
        self.dc.append(self.cube)
        self.slice_a = PathSlicedData(
            self.cube,
            self.cube.pixel_component_ids[1], [0., 1., 2.],
            self.cube.pixel_component_ids[2], [0., 1., 2.])
        self.slice_b = PathSlicedData(
            self.cube,
            self.cube.pixel_component_ids[1], [0., 1., 2., 3.],
            self.cube.pixel_component_ids[2], [0., 1., 2., 3.])
        self.dc.append(self.slice_a)
        self.dc.append(self.slice_b)

    def test_false_when_unrelated(self):
        # An LinkSame between a PV and its parent cube must not be
        # mistaken for a path link.
        link_path_sliced_to_parent(self.dc, self.slice_a)
        assert not path_link_exists(self.dc, self.slice_a, self.slice_b)

    def test_true_after_pair_link(self):
        link_path_sliced_pair_paths(self.dc, self.slice_a, self.slice_b)
        assert path_link_exists(self.dc, self.slice_a, self.slice_b)


class TestBuildOrUpdatePathSlices:
    """``build_or_update_path_slices`` (uses SimpleImageViewer)."""


    def setup_method(self, method):
        self.app, self.cube, self.viewer = _make_app_with_cube_viewer()
        self.dc = self.app.data_collection

    def test_creates_new_path_slice(self):
        updated = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])
        assert len(updated) == 1
        path_slice, _layer_state = updated[0]
        assert isinstance(path_slice, PathSlicedData)
        assert path_slice.original_data is self.cube
        assert path_slice.parent_viewer is self.viewer
        assert path_slice in self.dc

    def test_creates_parent_links(self):
        # After build_or_update, the new path slice is linked to its parent
        # cube's non-sliced axes (one LinkSame here -- axis 0).
        build_or_update_path_slices(self.viewer, [1, 2, 3], [0, 1, 2])
        assert len(self.dc.external_links) == 1

    def test_reuses_existing_slice(self):
        first = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])[0][0]
        first_x = first.x.copy()
        updated = build_or_update_path_slices(
            self.viewer, [0, 5, 2], [4, 0, 3])
        assert len(updated) == 1
        assert updated[0][0] is first
        assert not np.array_equal(first_x, updated[0][0].x)
        # sliced_dims must update too, in case the user changed x_att /
        # y_att in the source viewer between traces.
        assert updated[0][0].sliced_dims == (self.viewer.state.x_att.axis,
                                             self.viewer.state.y_att.axis)

    def test_pair_link_added_for_two_cubes(self):
        # Add a second cube to the viewer and verify a pairwise path link
        # is registered between the two new path slices.
        cube2 = _make_cube(label='cube2')
        self.dc.append(cube2)
        self.viewer.add_data(cube2)
        updated = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])
        assert len(updated) == 2
        slice_a, slice_b = updated[0][0], updated[1][0]
        assert path_link_exists(self.dc, slice_a, slice_b)

    def test_skips_subset_layers(self):
        # When the cube has a subset, the image viewer carries a
        # separate layer for it; its .layer is a Subset, not a Data,
        # and build_or_update_path_slices must skip it.
        self.dc.new_subset_group(label='s')
        assert len(self.viewer.state.layers) >= 2
        updated = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])
        # Only the cube produced a PV; the subset layer was ignored.
        assert len(updated) == 1

    def test_pair_link_not_re_added_on_re_extraction(self):
        cube2 = _make_cube(label='cube2')
        self.dc.append(cube2)
        self.viewer.add_data(cube2)
        build_or_update_path_slices(self.viewer, [1, 2, 3], [0, 1, 2])
        before = len(self.dc.external_links)
        build_or_update_path_slices(self.viewer, [0, 5, 2], [4, 0, 3])
        assert len(self.dc.external_links) == before


class TestOpenOrUpdateSliceViewer:
    """``open_or_update_slice_viewer`` (uses SimpleImageViewer)."""


    def setup_method(self, method):
        self.app, self.cube, self.viewer = _make_app_with_cube_viewer()
        self.dc = self.app.data_collection

    def test_opens_new_viewer_when_current_is_none(self):
        slice_viewer = open_or_update_slice_viewer(
            self.viewer, None, SimpleImageViewer,
            vx=[1, 2, 3], vy=[0, 1, 2])
        assert isinstance(slice_viewer, SimpleImageViewer)
        # A PathSlicedData was created and is now a layer of the new viewer.
        layers = [ls.layer for ls in slice_viewer.state.layers]
        path_slices = [d for d in layers if isinstance(d, PathSlicedData)]
        assert len(path_slices) == 1
        # The slice viewer auto-pinned the layer state for the PV.
        assert slice_viewer.state.aspect == 'auto'

    def test_refreshes_existing(self):
        # First call opens a new slice viewer.
        slice_viewer = open_or_update_slice_viewer(
            self.viewer, None, SimpleImageViewer,
            vx=[1, 2, 3], vy=[0, 1, 2])
        path_slice = next(d for d in self.dc if isinstance(d, PathSlicedData))
        first_x = path_slice.x.copy()
        # Second call with the same slice viewer reuses it and refreshes
        # the existing PathSlicedData in place (no new viewer opened).
        result = open_or_update_slice_viewer(
            self.viewer, slice_viewer, SimpleImageViewer,
            vx=[0, 5, 2], vy=[3, 4, 0])
        assert result is slice_viewer
        assert not np.array_equal(first_x, path_slice.x)

    def test_update_from_dict_value_error_swallowed(self):
        # If a SelectionCallbackProperty's choices aren't yet populated,
        # update_from_dict raises ValueError; the helper must continue.
        # Force the situation by patching update_from_dict on every new
        # layer state added to the slice viewer.
        original_add_data = SimpleImageViewer.add_data

        def add_data_with_bad_layer_state(self, data):
            original_add_data(self, data)
            # Make the most recently added layer state raise on update_from_dict.
            self.state.layers[-1].update_from_dict = MagicMock(
                side_effect=ValueError('not in choices'))

        SimpleImageViewer.add_data = add_data_with_bad_layer_state
        try:
            # No exception should bubble out.
            result = open_or_update_slice_viewer(
                self.viewer, None, SimpleImageViewer,
                vx=[1, 2, 3], vy=[0, 1, 2])
            assert isinstance(result, SimpleImageViewer)
        finally:
            SimpleImageViewer.add_data = original_add_data


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
