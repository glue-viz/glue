"""
Wire a group of :class:`PathSlicedData` instances into glue's
``ComponentLink`` graph so the generic
:func:`~glue.core.fixed_resolution_buffer.compute_fixed_resolution_buffer`
function can translate between them.

Typical use case: an image viewer showing an RGB composite of several
cubes linked by ``WCSLink`` (or any other cube-to-cube links). The
user draws a single path; one :class:`PathSlicedData` is created per
cube to expose the values along it, and these helpers wire the PVs up
so a path-slice viewer can render them together.

The hookup has two parts:

1. **Per-PV "self" links** between each PV's non-path pixel CIDs and
   the corresponding non-sliced pixel CIDs of its parent cube. The
   parent cubes are already linked to each other (via WCSLink, etc.)
   so the chain
   ``pv_a.non_path -> pv_a.original_data.non_sliced -> ... -> pv_b.original_data.non_sliced -> pv_b.non_path``
   is then walkable.

2. **Pairwise path links** between every pair of PVs: a
   :class:`PathRelativeLink` that maps path-axis pixel ``k`` in one PV
   to ``k * N_other / N_self`` in the other (the "same relative offset
   along the path" assumption that callers are expected to enforce).

Path lengths are read on every call inside :class:`PathRelativeLink`,
so :meth:`PathSlicedData.set_xy` updates propagate without any link
re-registration.
"""

import numpy as np

from glue.core.component_link import ComponentLink
from glue.core.link_helpers import LinkSame

from .path_sliced_data import PathSlicedData

__all__ = ['PathRelativeLink', 'link_path_sliced_to_parent',
           'link_path_sliced_pair_paths', 'link_path_sliced_group']


class PathRelativeLink(ComponentLink):
    """
    Link the path-axis pixel CIDs of two :class:`PathSlicedData` instances
    by relative offset along arc length.

    The forward mapping is ``k_to = k_from * len(pv_to.x) / len(pv_from.x)``.
    Path lengths are read on each call, so the link automatically reflects
    the current state of both PVs even after :meth:`PathSlicedData.set_xy`
    has resampled one of them.
    """

    def __init__(self, pv_from, pv_to):
        if not isinstance(pv_from, PathSlicedData) or not isinstance(pv_to, PathSlicedData):
            raise TypeError("PathRelativeLink requires two PathSlicedData instances")
        self._pv_from = pv_from
        self._pv_to = pv_to
        super().__init__(
            [pv_from.pixel_component_ids[-1]],
            pv_to.pixel_component_ids[-1],
            using=self._forward,
            inverse=self._backward,
            input_names=['k'],
            output_name='k',
        )

    def _forward(self, k_from):
        n_from = len(self._pv_from.x)
        n_to = len(self._pv_to.x)
        return np.asarray(k_from) * (n_to / n_from)

    def _backward(self, k_to):
        n_from = len(self._pv_from.x)
        n_to = len(self._pv_to.x)
        return np.asarray(k_to) * (n_from / n_to)


def link_path_sliced_to_parent(data_collection, pv):
    """
    Link a PathSlicedData's non-path pixel axes to the corresponding
    pixel axes of its parent cube via :class:`LinkSame`.

    The path axis is intentionally left unlinked: there is no single
    parent component that corresponds to the path index. Use
    :func:`link_path_sliced_pair_paths` (or
    :func:`link_path_sliced_group`) to link the path axes of PV pairs.

    Returns
    -------
    links : list
        The newly created link objects, in original-cube axis order.
    """
    non_sliced_axes = [a for a in range(pv.original_data.ndim)
                       if a not in pv.sliced_dims]
    links = []
    for pv_axis, parent_axis in enumerate(non_sliced_axes):
        link = LinkSame(pv.pixel_component_ids[pv_axis],
                        pv.original_data.pixel_component_ids[parent_axis])
        data_collection.add_link(link)
        links.append(link)
    return links


def link_path_sliced_pair_paths(data_collection, pv_a, pv_b):
    """
    Register a :class:`PathRelativeLink` between two PVs' path axes.

    Caller-side assumption: the two PVs trace the *same* underlying path
    (typically expressed in two cubes' pixel frames via a shared world-
    coord path). Their parent cubes' ``sliced_dims`` need not match --
    e.g. two cubes with the same physical axes in different storage
    order will produce PVs with different ``sliced_dims`` but the same
    non-path axes connect up through the cube-to-cube link graph.
    """
    link = PathRelativeLink(pv_a, pv_b)
    data_collection.add_link(link)
    return link


def link_path_sliced_group(data_collection, *pvs):
    """
    Wire a group of PathSlicedData instances into the link graph.

    For each PV, the non-path pixel axes are linked to the parent cube
    (via :func:`link_path_sliced_to_parent`). For every pair, the path
    axes are linked (via :func:`link_path_sliced_pair_paths`). With the
    parent cubes themselves already linked (e.g. by WCSLink), this is
    enough for the generic FRB function to translate between any two of
    the PVs in the group.

    Parameters
    ----------
    data_collection : :class:`~glue.core.DataCollection`
        The collection that owns all the PVs and their parent cubes.
    *pvs : :class:`PathSlicedData`
        Two or more PVs to wire up. Each must already be a member of
        ``data_collection``.

    Returns
    -------
    links : list
        Every link object created, in registration order.
    """
    if len(pvs) < 2:
        raise ValueError("link_path_sliced_group needs at least two PVs")
    links = []
    for pv in pvs:
        links.extend(link_path_sliced_to_parent(data_collection, pv))
    for i, pv_a in enumerate(pvs):
        for pv_b in pvs[i + 1:]:
            links.append(link_path_sliced_pair_paths(
                data_collection, pv_a, pv_b))
    return links
