"""
Backend-agnostic helpers for the path slicer plugin.

These functions are shared across the Qt and Jupyter front-ends; both
back-ends do the same data-model work (creating/updating a
:class:`PathSlicedData` per Data layer, wiring the link graph, opening
or refreshing the slice viewer, and driving the parent viewer's slice
index) and only differ in the viewer class they hand to
``new_data_viewer``.
"""
import numpy as np

from glue.core import Data
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData
from glue.plugins.tools.path_slicer.path_sliced_data_links import (
    link_path_sliced_to_parent, link_path_sliced_pair_paths)


__all__ = ['build_or_update_path_slices', 'path_link_exists',
           'drive_parent_slice', 'find_existing_path_slice',
           'open_or_update_slice_viewer',
           'create_trace', 'update_trace', 'open_slice_viewer_for']


def find_existing_path_slice(data_collection, parent_data):
    """Return the existing :class:`PathSlicedData` over ``parent_data``
    in ``data_collection``, or ``None`` if there is none."""
    for d in data_collection:
        if isinstance(d, PathSlicedData) and d.original_data is parent_data:
            return d
    return None


def path_link_exists(data_collection, slice_a, slice_b):
    """True if the link graph already has a link whose ends include the
    path-axis pixel CIDs of both slices."""
    cid_a = slice_a.pixel_component_ids[-1]
    cid_b = slice_b.pixel_component_ids[-1]
    for link in data_collection.external_links:
        ends = list(getattr(link, '_from', []))
        to = getattr(link, '_to', None)
        if to is not None:
            ends.append(to)
        if cid_a in ends and cid_b in ends:
            return True
    return False


def build_or_update_path_slices(source_viewer, vx, vy):
    """
    For each Data layer in ``source_viewer``, create (or update in
    place) the corresponding :class:`PathSlicedData`, register the
    needed ComponentLinks, and return the list of
    ``(path_slice, source_layer_state)`` pairs in iteration order.

    The caller is responsible for opening / reusing a slice viewer and
    populating it with the returned slices; this helper does only the
    data-model side of the work.
    """
    dc = source_viewer.session.data_collection
    x_att = source_viewer.state.x_att
    y_att = source_viewer.state.y_att

    updated = []
    for layer_state in source_viewer.state.layers:
        data = layer_state.layer
        if not isinstance(data, Data):
            # Subsets ride along with their parent Data.
            continue

        existing = find_existing_path_slice(dc, data)
        if existing is None:
            path_slice = PathSlicedData(data, x_att, vx, y_att, vy,
                                label=data.label + ' [slice]')
            path_slice.parent_viewer = source_viewer
            dc.append(path_slice)
            link_path_sliced_to_parent(dc, path_slice)
        else:
            path_slice = existing
            path_slice.cid_x = x_att
            path_slice.cid_y = y_att
            path_slice.sliced_dims = (x_att.axis, y_att.axis)
            path_slice.set_xy(vx, vy)
        updated.append((path_slice, layer_state))

    for i, (slice_a, _) in enumerate(updated):
        for slice_b, _ in updated[i + 1:]:
            if not path_link_exists(dc, slice_a, slice_b):
                link_path_sliced_pair_paths(dc, slice_a, slice_b)

    return updated


def open_or_update_slice_viewer(source_viewer, current_slice_viewer, slice_viewer_cls,
                             vx, vy):
    """
    Maintain the slice viewer attached to a path-slicer tool.

    If ``current_slice_viewer`` is ``None``, opens a fresh viewer of
    ``slice_viewer_cls``, materialises a :class:`PathSlicedData` per Data
    layer of ``source_viewer``, populates it, and copies layer visual
    state (colour, attribute, etc.) from the source viewer. Otherwise
    just refreshes the existing slices in place.

    The visual-state copy is best-effort: a not-yet-populated
    :class:`SelectionCallbackProperty` raises :class:`ValueError`, in
    which case that property is left at its default.

    Returns the (new or existing) slice viewer.
    """
    if current_slice_viewer is not None:
        build_or_update_path_slices(source_viewer, vx, vy)
        return current_slice_viewer

    slice_viewer = source_viewer.session.application.new_data_viewer(slice_viewer_cls)
    for path_slice, layer_state in build_or_update_path_slices(source_viewer, vx, vy):
        slice_viewer.add_data(path_slice)
        layer_attrs = layer_state.as_dict()
        layer_attrs.pop('layer', None)
        for new_layer_state in slice_viewer.state.layers[::-1]:
            if new_layer_state.layer is path_slice:
                try:
                    new_layer_state.update_from_dict(layer_attrs)
                except ValueError:
                    pass
                break
    slice_viewer.state.aspect = 'auto'
    if hasattr(slice_viewer.state, 'color_mode'):
        slice_viewer.state.color_mode = source_viewer.state.color_mode
    slice_viewer.state.reset_limits()
    return slice_viewer


def drive_parent_slice(path_slice, slice_y):
    """
    Push ``slice_y`` onto the parent viewer's slice index. Backend-
    agnostic: writes to ``ImageViewerState.slices``, which all image
    viewer back-ends share.

    Parameters
    ----------
    path_slice : :class:`PathSlicedData`
        The slice whose ``parent_viewer`` should have its slice updated.
    slice_y : float
        The y-coordinate in the slice viewer's frame -- a pixel index on
        the cube's non-sliced axis.
    """
    parent_viewer = path_slice.parent_viewer
    state = parent_viewer.state
    slc = list(state.slices)
    for i in range(state.reference_data.ndim):
        if i != state.x_att.axis and i != state.y_att.axis:
            slc[i] = int(slice_y)
    state.slices = tuple(slc)


def create_trace(source_viewer, vx, vy, existing_traces=()):
    """
    Materialise a fresh trace: one :class:`PathSlicedData` per Data
    layer in ``source_viewer``, all sharing the path ``(vx, vy)``.
    A "trace" is one Enter on the path tool; the caller keeps the
    list of traces and decides between create-new vs update-existing.

    The new PVs are appended to ``source_viewer.session.data_collection``,
    pairwise-linked against each other and against every PV in
    ``existing_traces``, and per-PV ``LinkSame`` registered against
    their parent cubes.

    Parameters
    ----------
    source_viewer
        Image viewer drawing the path.
    vx, vy : array-like
        Path vertices in the source viewer's pixel frame.
    existing_traces : iterable of list[PathSlicedData], optional
        The traces already produced by the calling tool. Pair-links
        are registered between every PV in the new trace and every PV
        in these existing traces, so the slice viewer can render them
        together if desired.

    Returns
    -------
    new_trace : list[PathSlicedData]
        The newly-created PVs, in the same order as ``source_viewer``
        iterates its Data layers.
    """
    dc = source_viewer.session.data_collection
    x_att = source_viewer.state.x_att
    y_att = source_viewer.state.y_att
    new_trace = []
    for layer_state in source_viewer.state.layers:
        data = layer_state.layer
        if not isinstance(data, Data):
            continue
        n_existing = sum(
            1 for trace in existing_traces for ps in trace
            if ps.original_data is data)
        label = f'{data.label} [slice {n_existing + 1}]'
        path = PathSlicedData(data, x_att, np.asarray(vx, dtype=float),
                              y_att, np.asarray(vy, dtype=float),
                              label=label)
        path.parent_viewer = source_viewer
        dc.append(path)
        link_path_sliced_to_parent(dc, path)
        new_trace.append(path)
    # Pairwise links between the new PVs and every existing PV...
    for trace in existing_traces:
        for slice_a in new_trace:
            for slice_b in trace:
                link_path_sliced_pair_paths(dc, slice_a, slice_b)
    # ...and between PVs within the new trace.
    for i, slice_a in enumerate(new_trace):
        for slice_b in new_trace[i + 1:]:
            link_path_sliced_pair_paths(dc, slice_a, slice_b)
    return new_trace


def update_trace(trace, vx, vy):
    """Refresh every :class:`PathSlicedData` in ``trace`` with a new
    ``(vx, vy)`` path. ``set_xy`` broadcasts a
    :class:`NumericalDataChangedMessage` so any viewer showing these
    PVs auto-refreshes."""
    vx_arr = np.asarray(vx, dtype=float)
    vy_arr = np.asarray(vy, dtype=float)
    for path in trace:
        path.set_xy(vx_arr, vy_arr)


def open_slice_viewer_for(source_viewer, slice_viewer_cls, paths):
    """
    Open a fresh viewer of ``slice_viewer_cls`` and populate it with
    ``paths`` (the PVs from a single newly-created trace). Visual state
    (aspect, color mode) is copied from ``source_viewer`` where it
    applies.
    """
    slice_viewer = source_viewer.session.application.new_data_viewer(
        slice_viewer_cls)
    for path in paths:
        slice_viewer.add_data(path)
    slice_viewer.state.aspect = 'auto'
    if hasattr(slice_viewer.state, 'color_mode'):
        slice_viewer.state.color_mode = source_viewer.state.color_mode
    slice_viewer.state.reset_limits()
    return slice_viewer
