import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glue.core import Data, DataCollection
from glue.core.coordinates import AffineCoordinates, IdentityCoordinates
from glue.core.data import BaseCartesianData
from glue.core import fixed_resolution_buffer as frb_mod
from glue.core.hub import HubListener
from glue.core.message import NumericalDataChangedMessage
from glue.core.link_helpers import LinkSame

from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData, sample_points
from glue.plugins.tools.path_slicer.path_sliced_data_links import link_path_sliced_group


# ---------------------------------------------------------------------------
# sample_points -- the pure helper. Test in isolation first.
# ---------------------------------------------------------------------------


class TestSamplePoints:

    def test_uniform_horizontal(self):
        # A straight horizontal path from x=0 to x=10 has length 10.
        # With spacing=1 we expect 11 samples (n=10, n+1=11) at y=0.
        x, y = sample_points([0., 10.], [0., 0.], spacing=1.)
        assert x.shape == (11,)
        assert_allclose(x, np.arange(11))
        assert_allclose(y, np.zeros(11))

    def test_uniform_diagonal(self):
        # 45-degree diagonal of length sqrt(2) * 10. With spacing=1 we get
        # floor(10*sqrt(2)) + 1 == 15 samples.
        x, y = sample_points([0., 10.], [0., 10.], spacing=1.)
        n = int(np.floor(10 * np.sqrt(2))) + 1
        assert x.size == n
        # Samples are equally spaced along the arc.
        dist = np.hypot(np.diff(x), np.diff(y))
        assert_allclose(dist, np.ones(dist.size))

    def test_multi_segment(self):
        # L-shape: (0,0) -> (3,0) -> (3,4). Length 7, spacing=1 -> 8 samples.
        x, y = sample_points([0., 3., 3.], [0., 0., 4.], spacing=1.)
        assert x.size == 8
        # First three samples are along the horizontal segment.
        assert_allclose(x[:4], [0, 1, 2, 3])
        assert_allclose(y[:4], [0, 0, 0, 0])
        # The remaining four are along the vertical segment at x=3.
        assert_allclose(x[4:], [3, 3, 3, 3])
        assert_allclose(y[4:], [1, 2, 3, 4])

    def test_spacing_changes_count(self):
        x, _ = sample_points([0., 10.], [0., 0.], spacing=2.)
        # n = floor(10 / 2) = 5 -> 6 samples
        assert x.size == 6
        assert_allclose(x, [0, 2, 4, 6, 8, 10])

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match='same shape'):
            sample_points([0., 1.], [0., 1., 2.])

    def test_not_1d(self):
        with pytest.raises(ValueError, match='1-d'):
            sample_points([[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]])

    def test_too_few_vertices(self):
        with pytest.raises(ValueError, match='at least two'):
            sample_points([1.], [1.])

    def test_nonpositive_spacing(self):
        with pytest.raises(ValueError, match='positive'):
            sample_points([0., 1.], [0., 0.], spacing=0)
        with pytest.raises(ValueError, match='positive'):
            sample_points([0., 1.], [0., 0.], spacing=-1)

    def test_path_shorter_than_spacing(self):
        with pytest.raises(ValueError, match='shorter than spacing'):
            sample_points([0., 0.5], [0., 0.], spacing=1.)


# ---------------------------------------------------------------------------
# PathSlicedData -- construction, basic properties, get_data, get_mask,
# compute_statistic, set_xy invalidations and messages, spacing parameter,
# error handling.
# ---------------------------------------------------------------------------


class _MessageRecorder(HubListener):
    """Collects ``NumericalDataChangedMessage`` instances for assertions."""

    def __init__(self):
        self.received = []

    def register_to_hub(self, hub):
        hub.subscribe(self, NumericalDataChangedMessage,
                      handler=self.received.append)


@pytest.fixture
def cube_dc():
    """A 3-d parent dataset (shape (6, 5, 4)) inside a DataCollection."""
    parent = Data(x=np.arange(120).reshape((6, 5, 4)),
                  coords=IdentityCoordinates(n_dim=3),
                  label='parent')
    dc = DataCollection([parent])
    return parent, dc


class TestPathSlicedDataConstruction:

    def test_basic_shape(self, cube_dc):
        parent, _ = cube_dc
        # Slice axes 1 and 2 with a 3-vertex path.
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [1., 2., 3.],
                          parent.pixel_component_ids[2], [0., 2., 5.])
        # Axis 0 of parent kept; axes 1+2 replaced by the path.
        assert path_slice.shape[0] == 6
        assert path_slice.ndim == 2
        # Path is resampled by `sample_points` with default spacing=1.
        # The original path total length is hypot(1,2)+hypot(1,3) ~= 5.4
        # so n=5 -> 6 samples.
        assert path_slice.shape[-1] == 6

    def test_label_default(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [1., 2.],
                          parent.pixel_component_ids[2], [0., 3.])
        assert path_slice.label == ''

    def test_label_explicit(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [1., 2.],
                          parent.pixel_component_ids[2], [0., 3.],
                          label='slice')
        assert path_slice.label == 'slice'

    def test_spacing_parameter(self, cube_dc):
        parent, _ = cube_dc
        # spacing=2 should halve the number of samples for the same path
        # compared to spacing=1.
        slice1 = PathSlicedData(parent,
                           parent.pixel_component_ids[1], [0., 4.],
                           parent.pixel_component_ids[2], [0., 0.],
                           spacing=1)
        slice2 = PathSlicedData(parent,
                           parent.pixel_component_ids[1], [0., 4.],
                           parent.pixel_component_ids[2], [0., 0.],
                           spacing=2)
        assert slice1.shape[-1] == 5
        assert slice2.shape[-1] == 3

    def test_same_axis_rejected(self, cube_dc):
        parent, _ = cube_dc
        with pytest.raises(ValueError, match='different axes'):
            PathSlicedData(parent,
                         parent.pixel_component_ids[1], [0., 2.],
                         parent.pixel_component_ids[1], [0., 2.])

    def test_propagates_sample_points_errors(self, cube_dc):
        parent, _ = cube_dc
        with pytest.raises(ValueError, match='same shape'):
            PathSlicedData(parent,
                         parent.pixel_component_ids[1], [0., 1.],
                         parent.pixel_component_ids[2], [0., 1., 2.])

    def test_main_components_delegates(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 2.])
        assert path_slice.main_components == parent.main_components

    def test_get_kind_delegates(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 2.])
        cid = parent.main_components[0]
        assert path_slice.get_kind(cid) == parent.get_kind(cid)


class TestPathSlicedDataAccessors:

    @pytest.fixture
    def path_slice(self, cube_dc):
        parent, _ = cube_dc
        return PathSlicedData(parent,
                            parent.pixel_component_ids[1], [0., 4.],
                            parent.pixel_component_ids[2], [0., 0.],
                            label='slice')

    def test_get_data_matches_manual_indexing(self, path_slice, cube_dc):
        parent, _ = cube_dc
        values = path_slice.get_data(parent.main_components[0])
        # Path is along axis 1 at y=0 of parent, so for each parent axis-0
        # index, expected[i, k] == parent[i, k, 0].
        expected = parent['x'][:, :path_slice.shape[-1], 0]
        assert values.shape == path_slice.shape
        assert_array_equal(values, expected)

    def test_get_data_pixel_cid_uses_super(self, path_slice):
        # Pixel component IDs are auto-generated by the base class; they
        # should not go through original_data.get_data.
        pix_cid = path_slice.pixel_component_ids[0]
        pix_values = path_slice.get_data(pix_cid)
        # Axis-0 pixel coords across the slice shape (6, n_path).
        assert pix_values.shape == path_slice.shape
        # Constant along the path axis, increasing along axis 0.
        assert_array_equal(pix_values[:, 0], np.arange(path_slice.shape[0]))

    def test_get_data_out_of_bounds_filled_with_zero(self, cube_dc):
        # A path that runs off the edge of the parent should produce zero
        # for the out-of-bounds samples instead of raising.
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 8.],  # axis 1 only has size 5
                          parent.pixel_component_ids[2], [0., 0.])
        values = path_slice.get_data(parent.main_components[0])
        # The first samples are in range, later ones go off the edge.
        in_range = values[0, :5]
        out_of_range = values[0, 5:]
        assert np.any(in_range != 0)
        assert_array_equal(out_of_range, np.zeros_like(out_of_range))

    def test_get_mask(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        # subset of parent: select where x >= 60. In flattened pixel-order
        # 120 cells, indices 60..119 satisfy.
        subset_state = parent.id['x'] >= 60
        mask = path_slice.get_mask(subset_state)
        assert mask.shape == path_slice.shape
        # Along axis-0 indices 3..5 of parent, x>=60 (since 3*20=60).
        assert_array_equal(mask[:3, :], np.zeros((3, path_slice.shape[-1])))
        assert_array_equal(mask[3:, :], np.ones((3, path_slice.shape[-1])))

    def test_get_mask_on_own_pixel_components(self, cube_dc):
        # Regression: a subset drawn on the slice viewer (i.e. expressed
        # against PathSlicedData's own pixel CIDs, not the parent
        # cube's) used to raise IncompatibleAttribute because get_mask
        # blindly delegated to original_data. The slice viewer's
        # subset overlay therefore never rendered.
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        # Range subset on the path axis: indices 1..3 inclusive.
        path_cid = path_slice.pixel_component_ids[-1]
        subset_state = (path_cid >= 1) & (path_cid <= 3)
        mask = path_slice.get_mask(subset_state)
        assert mask.shape == path_slice.shape
        # Path axis is the last axis; the selection should be True for
        # path indices in [1, 3] regardless of the non-path axis.
        expected = np.zeros(path_slice.shape, dtype=bool)
        expected[:, 1:4] = True
        assert_array_equal(mask, expected)

    def test_compute_statistic_along_path(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        # Sum along axis 0 of path_slice = sum over parent axis 0 of the chosen path
        # cells.
        result = path_slice.compute_statistic('sum', parent.main_components[0], axis=0)
        # For each path sample k, the path lands on (axis1=int(round(x[k])),
        # axis2=0) of the parent. So expected[k] = sum_i parent[i, x[k], 0].
        assert result.shape == (path_slice.shape[-1],)
        # The path has y=0 throughout and x integer values 0..4. So:
        x_int = np.round(path_slice.x).astype(int)
        expected = np.array([parent['x'][:, ix, 0].sum() for ix in x_int])
        assert_array_equal(result, expected)

    def test_compute_statistic_scalar_no_axis(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        # No axis argument -> scalar reduction over the whole slice array.
        total = path_slice.compute_statistic('sum', parent.main_components[0])
        assert np.isscalar(total) or total.shape == ()
        # Independent computation via get_data must match.
        assert total == path_slice.get_data(parent.main_components[0]).sum()

    def test_compute_statistic_with_subset_state(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        # Sum, but masked to where x >= 60 on the parent.
        result = path_slice.compute_statistic('sum', parent.main_components[0],
                                      subset_state=parent.id['x'] >= 60)
        # The first three slice rows (parent axis 0 indices 0..2) are excluded
        # entirely; the last three contribute fully.
        values = path_slice.get_data(parent.main_components[0])
        mask = path_slice.get_mask(parent.id['x'] >= 60)
        assert result == values[mask].sum()

    def test_get_data_with_array_view(self, cube_dc):
        # Exercise the advanced-indexing branch of _get_pix_coords:
        # callers pass a tuple of 1-d ndarrays (same length) and get back
        # a 1-d result. This is the path FRB uses internally.
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        i_view = np.array([0, 2, 5])      # along path_slice axis 0
        k_view = np.array([1, 2, 4])      # along path axis
        values = path_slice.get_data(parent.main_components[0], view=(i_view, k_view))
        assert values.shape == (3,)
        # Each value should equal parent[i, x[k], y[k]] for the chosen pair.
        x_int = np.round(path_slice.x).astype(int)
        y_int = np.round(path_slice.y).astype(int)
        expected = np.array([parent['x'][i, x_int[k], y_int[k]]
                             for i, k in zip(i_view, k_view)])
        assert_array_equal(values, expected)

    def test_compute_histogram_delegates(self, cube_dc):
        # compute_histogram passes through to the parent; just confirm
        # the call doesn't transform the args.
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 4.],
                          parent.pixel_component_ids[2], [0., 0.])
        cid = parent.main_components[0]
        # Compare against the parent's own histogram.
        h_pv = path_slice.compute_histogram([cid], range=[(0, 120)], bins=[12])
        h_parent = parent.compute_histogram([cid], range=[(0, 120)], bins=[12])
        assert_array_equal(h_pv, h_parent)


class TestSetXY:

    def test_replaces_path_and_shape(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 0.])
        old_len = path_slice.shape[-1]
        path_slice.set_xy([0., 4.], [0., 0.])
        assert path_slice.shape[-1] != old_len
        assert path_slice.shape[-1] == 5

    def test_broadcasts_message_after_attach(self, cube_dc):
        parent, dc = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 0.])
        # Attaching to a DataCollection wires up the hub.
        dc.append(path_slice)
        recorder = _MessageRecorder()
        recorder.register_to_hub(dc.hub)
        path_slice.set_xy([0., 4.], [0., 0.])
        senders = [m.sender for m in recorder.received]
        assert path_slice in senders

    def test_construction_does_not_broadcast_when_unattached(self, cube_dc):
        # Constructing a PathSlicedData on a parent that has no hub must not
        # raise even though set_xy is called in __init__.
        parent = Data(x=np.zeros((4, 4, 4)), label='nohub')
        # No DataCollection -> parent.hub is None.
        PathSlicedData(parent,
                     parent.pixel_component_ids[1], [0., 2.],
                     parent.pixel_component_ids[2], [0., 0.])

    def test_set_xy_propagates_validation_errors(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 0.])
        with pytest.raises(ValueError, match='same shape'):
            path_slice.set_xy([0., 1.], [0., 1., 2.])

    def test_set_xy_invalidates_referencing_caches(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 0.])
        # Inject a fake entry that references ``path_slice`` as ``data``.
        frb_mod.ARRAY_CACHE['unit-test-path_slice'] = {
            'hash': (path_slice, (0, 5, 6), None, None, True),
            'array': np.zeros((6,)),
        }
        frb_mod.PIXEL_CACHE['unit-test-path_slice'] = {'hash': (path_slice, None)}
        # And an unrelated entry that must survive.
        frb_mod.ARRAY_CACHE['unit-test-other'] = {
            'hash': (object(), (0, 1, 2), None, None, True),
            'array': np.zeros((1,)),
        }
        try:
            path_slice.set_xy([0., 4.], [0., 0.])
            assert 'unit-test-path_slice' not in frb_mod.ARRAY_CACHE
            assert 'unit-test-path_slice' not in frb_mod.PIXEL_CACHE
            assert 'unit-test-other' in frb_mod.ARRAY_CACHE
        finally:
            frb_mod.ARRAY_CACHE.pop('unit-test-path_slice', None)
            frb_mod.PIXEL_CACHE.pop('unit-test-path_slice', None)
            frb_mod.ARRAY_CACHE.pop('unit-test-other', None)

    def test_set_xy_invalidates_when_self_is_target(self, cube_dc):
        # Cache entries can also have ``path_slice`` in the target_data slot.
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 0.])
        frb_mod.ARRAY_CACHE['target-slot'] = {
            'hash': (object(), (0, 1, 2), path_slice, None, True),
            'array': np.zeros((1,)),
        }
        try:
            path_slice.set_xy([0., 4.], [0., 0.])
            assert 'target-slot' not in frb_mod.ARRAY_CACHE
        finally:
            frb_mod.ARRAY_CACHE.pop('target-slot', None)

    def test_set_xy_skips_entries_without_hash(self, cube_dc):
        parent, _ = cube_dc
        path_slice = PathSlicedData(parent,
                          parent.pixel_component_ids[1], [0., 2.],
                          parent.pixel_component_ids[2], [0., 0.])
        # Defensive: malformed cache entries shouldn't crash invalidation.
        frb_mod.ARRAY_CACHE['no-hash'] = {'array': np.zeros((1,))}
        try:
            path_slice.set_xy([0., 4.], [0., 0.])
            assert 'no-hash' in frb_mod.ARRAY_CACHE
        finally:
            frb_mod.ARRAY_CACHE.pop('no-hash', None)


class TestComputeFixedResolutionBuffer:

    def _make_linked_slice_pair(self):
        # Two parent datasets with identical coords, linked pixel-to-pixel,
        # and matching slice slices on both. Link the slices into the graph so
        # the generic FRB function can translate slice2 -> slice1.
        matrix = np.array([[2, 0, 0, 0], [0, 2, 0, 0],
                           [0, 0, 2, 0], [0, 0, 0, 1]])
        data1 = Data(x=np.arange(120).reshape((6, 5, 4)),
                     coords=AffineCoordinates(matrix), label='d1')
        data2 = Data(y=np.arange(120).reshape((6, 5, 4)),
                     coords=IdentityCoordinates(n_dim=3), label='d2')
        dc = DataCollection([data1, data2])
        for idim in range(3):
            dc.add_link(LinkSame(data1.world_component_ids[idim],
                                 data2.world_component_ids[idim]))
        x1 = [0, 2, 5]
        y1 = [1, 2, 3]
        slice1 = PathSlicedData(data1,
                           data1.pixel_component_ids[1], y1,
                           data1.pixel_component_ids[2], x1)
        x2, y2, _ = data2.coords.world_to_pixel_values(
            *data1.coords.pixel_to_world_values(x1, y1, 0))
        slice2 = PathSlicedData(data2,
                           data2.pixel_component_ids[1], y2,
                           data2.pixel_component_ids[2], x2)
        dc.append(slice1)
        dc.append(slice2)
        link_path_sliced_group(dc, slice1, slice2)
        return data1, data2, slice1, slice2

    def test_linked_pair_shape(self):
        _, _, slice1, slice2 = self._make_linked_slice_pair()
        result = slice1.compute_fixed_resolution_buffer(
            bounds=[(0, 5, 15), (0, 6, 20)],
            target_data=slice2,
            target_cid=slice1.original_data.id['x'])
        assert result.shape == (15, 20)


# ---------------------------------------------------------------------------
# Signature-pin tests. The PR's contract relies on cache_id being part of
# BaseCartesianData.compute_fixed_resolution_buffer; pinning it prevents
# silent regressions.
# ---------------------------------------------------------------------------


class TestBaseCartesianDataSignature:

    def test_compute_fixed_resolution_buffer_accepts_cache_id(self):
        params = inspect.signature(
            BaseCartesianData.compute_fixed_resolution_buffer).parameters
        assert 'cache_id' in params
        assert params['cache_id'].default is None

    def test_path_sliced_data_signature_matches_base(self):
        # PathSlicedData's overriding method must declare the same kwargs so
        # callers can swap one for the other without surprises.
        base_params = inspect.signature(
            BaseCartesianData.compute_fixed_resolution_buffer).parameters
        sub_params = inspect.signature(
            PathSlicedData.compute_fixed_resolution_buffer).parameters
        for name in base_params:
            assert name in sub_params, (
                f"PathSlicedData.compute_fixed_resolution_buffer is missing "
                f"parameter {name!r} present on BaseCartesianData")
