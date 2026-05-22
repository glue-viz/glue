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


__all__ = ['drive_parent_slice',
           'create_trace', 'update_trace', 'open_slice_viewer_for']


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
