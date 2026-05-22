"""
Tests for link-based inter-PathSlicedData coordinate translation.

The realistic scenario: an RGB-style slice viewer that overplots ``N``
PathSlicedData instances backed by ``N`` different cubes which are
themselves linked via world coordinates (e.g. ``WCSLink``). The tests
here use ``LinkSame`` on world-component CIDs to stand in for a WCS
link.
"""

import numpy as np
from numpy.testing import assert_allclose

from glue.core import Data, DataCollection
from glue.core.coordinates import AffineCoordinates, IdentityCoordinates
from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer
from glue.core.link_helpers import LinkSame

from ..path_sliced_data import PathSlicedData
from ..path_sliced_data_links import (PathRelativeLink,
                                      link_path_sliced_to_parent,
                                      link_path_sliced_group)


def _make_slice_pair_linked_cubes():
    """
    Two distinct 6x5x4 cubes linked pixel-to-pixel through world coords
    (the diagonal AffineCoordinates is the identity on pixel space, so
    pixel i in cube_a == pixel i in cube_b). Each cube gets one slice
    along the same parent axes.
    """
    matrix = np.eye(4)
    cube_a = Data(x=np.arange(120.).reshape((6, 5, 4)),
                  coords=AffineCoordinates(matrix), label='cube_a')
    cube_b = Data(y=10. * np.arange(120.).reshape((6, 5, 4)),
                  coords=IdentityCoordinates(n_dim=3), label='cube_b')
    dc = DataCollection([cube_a, cube_b])
    for idim in range(3):
        dc.add_link(LinkSame(cube_a.world_component_ids[idim],
                             cube_b.world_component_ids[idim]))
    slice_a = PathSlicedData(cube_a,
                          cube_a.pixel_component_ids[1], [1., 2., 3.],
                          cube_a.pixel_component_ids[2], [0., 1., 3.],
                          label='slice_a')
    slice_b = PathSlicedData(cube_b,
                          cube_b.pixel_component_ids[1], [0.5, 1.5, 2.5, 3.5],
                          cube_b.pixel_component_ids[2], [0., 1., 2., 3.],
                          label='slice_b')
    dc.append(slice_a)
    dc.append(slice_b)
    return dc, cube_a, cube_b, slice_a, slice_b


def test_link_path_sliced_to_parent_only_touches_non_path_axes():
    # The non-path axes get linked to their parent's non-sliced
    # axes; the path axis is left alone (no parent component to link
    # it to).
    dc, _, _, slice_a, _ = _make_slice_pair_linked_cubes()
    links = link_path_sliced_to_parent(dc, slice_a)
    # slice_a has one non-path axis -> one LinkSame.
    assert len(links) == 1


def test_path_link_picks_up_set_xy_without_recreation():
    _, _, _, slice_a, slice_b = _make_slice_pair_linked_cubes()
    link = PathRelativeLink(slice_a, slice_b)
    before = link._forward(np.array([1.0]))
    slice_a.set_xy([0., 5., 10., 15.], [0., 0., 0., 0.])
    after = link._forward(np.array([1.0]))
    assert not np.allclose(before, after)
    n_a, n_b = len(slice_a.x), len(slice_b.x)
    assert_allclose(after, np.array([1.0]) * (n_b / n_a))


def test_generic_frb_via_links_across_linked_cubes():
    # With the right links in place, the generic FRB function can walk
    # slice_b -> cube_b -> cube_a -> slice_a entirely through the link
    # graph, no PathSlicedData override required.
    dc, cube_a, _, slice_a, slice_b = _make_slice_pair_linked_cubes()
    link_path_sliced_group(dc, slice_a, slice_b)

    bounds = [(0, 5, 6), (0, 2, 3)]
    buffer = compute_fixed_resolution_buffer(
        slice_a, bounds=bounds, target_data=slice_b, target_cid=cube_a.id['x'])
    assert buffer.shape == (6, 3)

    # Independently compute the expected values: slice_b pixel k_b ->
    # PathRelativeLink -> slice_a pixel k_a (FRB rounds), then
    # PathSlicedData._get_pix_coords truncates slice_a.x / slice_a.y when
    # looking them up in cube_a.
    cube = cube_a['x']
    n_a, n_b = len(slice_a.x), len(slice_b.x)
    expected = np.empty((6, 3))
    for i in range(6):
        for k_b in range(3):
            k_a = round(k_b * n_a / n_b)
            expected[i, k_b] = cube[i,
                                    int(slice_a.x[k_a]),
                                    int(slice_a.y[k_a])]
    assert_allclose(buffer, expected)


def test_link_path_sliced_group_handles_three_pvs():
    # The RGB use case: three cubes, three slices, the same path drawn in
    # each cube. After link_path_sliced_group, any slice pair can act as
    # data/target_data.
    matrix = np.eye(4)
    cubes = []
    path_slices = []
    dc = DataCollection()
    for label, scale in (('r', 1.), ('g', 2.), ('b', 3.)):
        cube = Data(x=scale * np.arange(120.).reshape((6, 5, 4)),
                    coords=AffineCoordinates(matrix), label=f'cube_{label}')
        dc.append(cube)
        cubes.append(cube)
    for idim in range(3):
        for cube in cubes[1:]:
            dc.add_link(LinkSame(cubes[0].world_component_ids[idim],
                                 cube.world_component_ids[idim]))
    for cube, label in zip(cubes, 'rgb'):
        path_slice = PathSlicedData(cube,
                            cube.pixel_component_ids[1], [1., 2., 3.],
                            cube.pixel_component_ids[2], [0., 1., 3.],
                            label=f'slice_{label}')
        dc.append(path_slice)
        path_slices.append(path_slice)

    links = link_path_sliced_group(dc, *path_slices)
    # 3 slices * 1 non-path axis = 3 self-links, plus C(3, 2) = 3 path links.
    assert len(links) == 6

    # Every pair should be navigable via the link graph; spot-check by
    # asking for an FRB from each slice into each other slice and checking
    # the result has the right shape.
    bounds = [(0, 5, 6), (0, 2, 3)]
    for data_pv in path_slices:
        for target_pv in path_slices:
            if data_pv is target_pv:
                continue
            buf = compute_fixed_resolution_buffer(
                data_pv, bounds=bounds, target_data=target_pv,
                target_cid=data_pv.original_data.id['x'])
            assert buf.shape == (6, 3)
