"""
Tests for the backend-agnostic helpers in
:mod:`glue.plugins.tools.path_slicer.common`.

The viewer-facing helpers operate on an image-viewer-like object;
the tests here stub one out with :class:`unittest.mock.MagicMock`
plus real :class:`Data` / :class:`DataCollection` so the data-model
plumbing remains exercised. End-to-end tests against the real Qt and
Jupyter image viewers live in glue-qt and glue-jupyter.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from glue.core import Data, DataCollection
from glue.core.coordinates import IdentityCoordinates
from glue.plugins.tools.path_slicer.common import (
    build_or_update_path_slices, drive_parent_slice, find_existing_path_slice,
    open_or_update_slice_viewer, path_link_exists)
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData
from glue.plugins.tools.path_slicer.path_sliced_data_links import (
    link_path_sliced_pair_paths, link_path_sliced_to_parent)


def _make_cube(label='cube', shape=(6, 5, 4)):
    return Data(label=label,
                x=np.arange(np.prod(shape), dtype=float).reshape(shape),
                coords=IdentityCoordinates(n_dim=len(shape)))


def _make_source_viewer(cube, dc, *, color_mode='Colormaps'):
    """A MagicMock dressed up as the cube viewer that build_or_update
    helpers read from."""
    viewer = MagicMock()
    viewer.session.data_collection = dc
    viewer.state.x_att = cube.pixel_component_ids[1]
    viewer.state.y_att = cube.pixel_component_ids[2]
    viewer.state.color_mode = color_mode
    layer_state = MagicMock()
    layer_state.layer = cube
    layer_state.as_dict.return_value = {'layer': cube,
                                        'attribute': cube.main_components[0]}
    viewer.state.layers = [layer_state]
    return viewer


# ---------------------------------------------------------------------------
# find_existing_path_slice
# ---------------------------------------------------------------------------


class TestFindExistingPathSlice:

    def setup_method(self, method):
        self.dc = DataCollection()
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


# ---------------------------------------------------------------------------
# path_link_exists
# ---------------------------------------------------------------------------


class TestPathLinkExists:

    def setup_method(self, method):
        self.dc = DataCollection()
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
        # Adding an unrelated link should not be mistaken for a path link.
        link_path_sliced_to_parent(self.dc, self.slice_a)
        assert not path_link_exists(self.dc, self.slice_a, self.slice_b)

    def test_true_after_pair_link(self):
        link_path_sliced_pair_paths(self.dc, self.slice_a, self.slice_b)
        assert path_link_exists(self.dc, self.slice_a, self.slice_b)


# ---------------------------------------------------------------------------
# build_or_update_path_slices
# ---------------------------------------------------------------------------


class TestBuildOrUpdatePathSlices:

    def setup_method(self, method):
        self.dc = DataCollection()
        self.cube = _make_cube()
        self.dc.append(self.cube)
        self.viewer = _make_source_viewer(self.cube, self.dc)

    def test_creates_new_path_slice(self):
        updated = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])
        assert len(updated) == 1
        path_slice, layer_state = updated[0]
        assert isinstance(path_slice, PathSlicedData)
        assert path_slice.original_data is self.cube
        assert path_slice.parent_viewer is self.viewer
        assert path_slice in self.dc
        # The layer state returned is the one from the source viewer.
        assert layer_state is self.viewer.state.layers[0]

    def test_creates_parent_links(self):
        # After build_or_update, the new path slice is linked to its parent
        # cube's non-sliced axes (axis 0 here).
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
        # set_xy must have replaced the path.
        assert not np.array_equal(first_x, updated[0][0].x)
        # Updating in place must update sliced_dims (in case the user
        # switched x_att / y_att in the source viewer between traces).
        assert updated[0][0].sliced_dims == (self.viewer.state.x_att.axis,
                                             self.viewer.state.y_att.axis)

    def test_skips_non_data_layers(self):
        # A layer whose .layer isn't a Data instance (e.g. a Subset) must be
        # skipped: subsets ride along with their parent Data automatically.
        non_data_layer = MagicMock()
        non_data_layer.layer = MagicMock()  # not a Data
        self.viewer.state.layers.append(non_data_layer)
        updated = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])
        assert len(updated) == 1

    def test_pair_link_added_for_two_cubes(self):
        # Add a second cube + a second source layer and verify a pairwise
        # path link is registered between the two created path slices.
        cube2 = _make_cube(label='cube2')
        self.dc.append(cube2)
        layer2 = MagicMock()
        layer2.layer = cube2
        layer2.as_dict.return_value = {'layer': cube2,
                                       'attribute': cube2.main_components[0]}
        self.viewer.state.layers.append(layer2)
        updated = build_or_update_path_slices(
            self.viewer, [1, 2, 3], [0, 1, 2])
        assert len(updated) == 2
        slice_a, slice_b = updated[0][0], updated[1][0]
        assert path_link_exists(self.dc, slice_a, slice_b)

    def test_pair_link_not_re_added_on_re_extraction(self):
        # Two cubes -> 1 pair link. Re-extracting must not double the link.
        cube2 = _make_cube(label='cube2')
        self.dc.append(cube2)
        layer2 = MagicMock()
        layer2.layer = cube2
        layer2.as_dict.return_value = {'layer': cube2,
                                       'attribute': cube2.main_components[0]}
        self.viewer.state.layers.append(layer2)
        build_or_update_path_slices(self.viewer, [1, 2, 3], [0, 1, 2])
        before = len(self.dc.external_links)
        build_or_update_path_slices(self.viewer, [0, 5, 2], [4, 0, 3])
        assert len(self.dc.external_links) == before


# ---------------------------------------------------------------------------
# open_or_update_slice_viewer
# ---------------------------------------------------------------------------


class TestOpenOrUpdateSliceViewer:

    def setup_method(self, method):
        self.dc = DataCollection()
        self.cube = _make_cube()
        self.dc.append(self.cube)
        self.source_viewer = _make_source_viewer(self.cube, self.dc)

    def _make_slice_viewer(self):
        slice_viewer = MagicMock()
        slice_viewer.state.layers = []

        def add_data(data):
            layer = MagicMock()
            layer.layer = data
            slice_viewer.state.layers.append(layer)
        slice_viewer.add_data.side_effect = add_data
        return slice_viewer

    def test_opens_new_viewer_when_current_is_none(self):
        slice_viewer = self._make_slice_viewer()
        self.source_viewer.session.application.new_data_viewer.return_value = slice_viewer
        result = open_or_update_slice_viewer(
            self.source_viewer, None, MagicMock(),
            vx=[1, 2, 3], vy=[0, 1, 2])
        assert result is slice_viewer
        slice_viewer.add_data.assert_called_once()
        slice_viewer.state.reset_limits.assert_called_once()
        assert slice_viewer.state.aspect == 'auto'

    def test_copies_color_mode_when_supported(self):
        # The source viewer has color_mode set; verify it propagates.
        slice_viewer = self._make_slice_viewer()
        self.source_viewer.session.application.new_data_viewer.return_value = slice_viewer
        open_or_update_slice_viewer(
            self.source_viewer, None, MagicMock(),
            vx=[1, 2, 3], vy=[0, 1, 2])
        assert slice_viewer.state.color_mode == self.source_viewer.state.color_mode

    def test_update_from_dict_value_error_swallowed(self):
        # If a SelectionCallbackProperty's choices aren't yet populated,
        # update_from_dict raises ValueError; the helper must continue.
        slice_viewer = self._make_slice_viewer()

        def add_data_bad(data):
            layer = MagicMock()
            layer.layer = data
            layer.update_from_dict.side_effect = ValueError("not in choices")
            slice_viewer.state.layers.append(layer)
        slice_viewer.add_data.side_effect = add_data_bad
        self.source_viewer.session.application.new_data_viewer.return_value = slice_viewer
        # No exception should bubble out.
        result = open_or_update_slice_viewer(
            self.source_viewer, None, MagicMock(),
            vx=[1, 2, 3], vy=[0, 1, 2])
        assert result is slice_viewer
        slice_viewer.state.reset_limits.assert_called_once()

    def test_refreshes_existing(self):
        # Pre-create the path slice + link so the data side reuses it.
        path_slice = PathSlicedData(
            self.cube,
            self.cube.pixel_component_ids[1], [0., 1., 2.],
            self.cube.pixel_component_ids[2], [0., 1., 2.])
        self.dc.append(path_slice)
        link_path_sliced_to_parent(self.dc, path_slice)
        first_x = path_slice.x.copy()
        existing = MagicMock()
        result = open_or_update_slice_viewer(
            self.source_viewer, existing, MagicMock(),
            vx=[5, 0, 2], vy=[3, 4, 0])
        assert result is existing
        assert not np.array_equal(first_x, path_slice.x)
        # No new viewer was opened.
        self.source_viewer.session.application.new_data_viewer.assert_not_called()


# ---------------------------------------------------------------------------
# drive_parent_slice
# ---------------------------------------------------------------------------


class TestDriveParentSlice:

    def test_writes_to_non_displayed_axis(self):
        cube = _make_cube()
        parent_viewer = MagicMock()
        parent_viewer.state.reference_data = cube
        parent_viewer.state.x_att = cube.pixel_component_ids[1]
        parent_viewer.state.y_att = cube.pixel_component_ids[2]
        parent_viewer.state.slices = (0, 0, 0)
        path_slice = PathSlicedData(
            cube,
            cube.pixel_component_ids[1], [0., 1., 2.],
            cube.pixel_component_ids[2], [0., 1., 2.])
        path_slice.parent_viewer = parent_viewer
        drive_parent_slice(path_slice, 4.0)
        # The non-displayed axis (axis 0 here) is set to int(4).
        assert parent_viewer.state.slices == (4, 0, 0)

    def test_truncates_float_y_to_int(self):
        cube = _make_cube()
        parent_viewer = MagicMock()
        parent_viewer.state.reference_data = cube
        parent_viewer.state.x_att = cube.pixel_component_ids[0]
        parent_viewer.state.y_att = cube.pixel_component_ids[2]
        parent_viewer.state.slices = (0, 0, 0)
        path_slice = PathSlicedData(
            cube,
            cube.pixel_component_ids[0], [0., 1., 2.],
            cube.pixel_component_ids[2], [0., 1., 2.])
        path_slice.parent_viewer = parent_viewer
        drive_parent_slice(path_slice, 2.7)
        # int() truncates toward zero; axis 1 here.
        assert parent_viewer.state.slices == (0, 2, 0)


# ---------------------------------------------------------------------------
# Misc: validation errors that aren't covered through the happy path
# ---------------------------------------------------------------------------


class TestPathRelativeLinkValidation:

    def test_init_rejects_non_path_sliced_data(self):
        from glue.plugins.tools.path_slicer.path_sliced_data_links import (
            PathRelativeLink)
        cube = _make_cube()
        with pytest.raises(TypeError, match='PathSlicedData'):
            PathRelativeLink(cube, cube)


class TestLinkGroupValidation:

    def test_link_group_rejects_single_input(self):
        from glue.plugins.tools.path_slicer.path_sliced_data_links import (
            link_path_sliced_group)
        cube = _make_cube()
        dc = DataCollection([cube])
        path_slice = PathSlicedData(
            cube,
            cube.pixel_component_ids[1], [0., 1., 2.],
            cube.pixel_component_ids[2], [0., 1., 2.])
        dc.append(path_slice)
        with pytest.raises(ValueError, match='at least two'):
            link_path_sliced_group(dc, path_slice)
