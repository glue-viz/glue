"""
Backend-agnostic helpers for the path slicer plugin.

These functions are shared across the Qt and Jupyter front-ends; both
back-ends do the same data-model work (creating/updating a
:class:`PathSlicedData` per Data layer, wiring the link graph, and
driving the parent viewer's slice index) and only differ in the viewer
class they hand to ``new_data_viewer``.
"""
import numpy as np

from glue.core import Data
from glue.plugins.tools.path_slicer.path_sliced_data import PathSlicedData
from glue.plugins.tools.path_slicer.path_sliced_data_links import (
    link_path_sliced_to_parent, link_path_sliced_pair_paths)


__all__ = ['build_or_update_pvs', 'path_link_exists',
           'drive_parent_slice', 'find_existing_pv']


def find_existing_pv(data_collection, parent_data):
    """Return the existing :class:`PathSlicedData` over ``parent_data``
    in ``data_collection``, or ``None`` if there is none."""
    for d in data_collection:
        if isinstance(d, PathSlicedData) and d.original_data is parent_data:
            return d
    return None


def path_link_exists(data_collection, pv_a, pv_b):
    """True if the link graph already has a link whose ends include the
    path-axis pixel CIDs of both PVs."""
    cid_a = pv_a.pixel_component_ids[-1]
    cid_b = pv_b.pixel_component_ids[-1]
    for link in data_collection.external_links:
        ends = list(getattr(link, '_from', []))
        to = getattr(link, '_to', None)
        if to is not None:
            ends.append(to)
        if cid_a in ends and cid_b in ends:
            return True
    return False


def build_or_update_pvs(source_viewer, vx, vy):
    """
    For each Data layer in ``source_viewer``, create (or update in
    place) the corresponding :class:`PathSlicedData`, register the
    needed ComponentLinks, and return the list of
    ``(pv, source_layer_state)`` pairs in iteration order.

    The caller is responsible for opening / reusing a PV viewer and
    populating it with the returned PVs; this helper does only the
    data-model side of the work.
    """
    dc = source_viewer.session.data_collection
    x_att = source_viewer.state.x_att
    y_att = source_viewer.state.y_att
    vx = np.asarray(vx)
    vy = np.asarray(vy)

    updated = []
    for layer_state in source_viewer.state.layers:
        data = layer_state.layer
        if not isinstance(data, Data):
            # Subsets ride along with their parent Data.
            continue

        existing = find_existing_pv(dc, data)
        if existing is None:
            pv = PathSlicedData(data, x_att, vx, y_att, vy,
                                label=data.label + ' [slice]')
            pv.parent_viewer = source_viewer
            dc.append(pv)
            link_path_sliced_to_parent(dc, pv)
        else:
            pv = existing
            pv.cid_x = x_att
            pv.cid_y = y_att
            pv.sliced_dims = (x_att.axis, y_att.axis)
            pv.set_xy(vx, vy)
        updated.append((pv, layer_state))

    for i, (pv_a, _) in enumerate(updated):
        for pv_b, _ in updated[i + 1:]:
            if not path_link_exists(dc, pv_a, pv_b):
                link_path_sliced_pair_paths(dc, pv_a, pv_b)

    return updated


def drive_parent_slice(pv, pv_y_value):
    """
    Push ``pv_y_value`` onto the parent viewer's slice index. Backend-
    agnostic: writes to ``ImageViewerState.slices``, which all image
    viewer back-ends share.

    Parameters
    ----------
    pv : :class:`PathSlicedData`
        The PV whose ``parent_viewer`` should have its slice updated.
    pv_y_value : float
        The y-coordinate in the PV viewer's frame -- a pixel index on
        the cube's non-sliced axis.
    """
    parent_viewer = pv.parent_viewer
    state = parent_viewer.state
    slc = list(state.slices)
    for i in range(state.reference_data.ndim):
        if i != state.x_att.axis and i != state.y_att.axis:
            slc[i] = int(pv_y_value)
    state.slices = tuple(slc)
