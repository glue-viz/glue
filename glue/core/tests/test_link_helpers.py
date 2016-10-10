# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from glue.core import ComponentID, Data, Component, DataCollection

from .. import link_helpers as lh
from ..link_helpers import (LinkTwoWay, multi_link,
                            LinkSame, LinkAligned)


R, D, L, B = (ComponentID('ra'), ComponentID('dec'),
              ComponentID('lon'), ComponentID('lat'))


def forwards(x, y):
    print('forwads inputs', x, y)
    return x * 3, y * 5


def backwards(x, y):
    print('backwards inputs', x, y)
    return x / 3, y / 5


def check_link(link, from_, to, using=None):
    assert link.get_from_ids() == from_
    assert link.get_to_id() == to
    if using:
        assert link.get_using() == using


def check_using(link, inp, out):
    np.testing.assert_array_almost_equal(link.get_using()(*inp), out)


def test_LinkTwoWay():
    result = LinkTwoWay(R, D, forwards, backwards)
    check_link(result[0], [R], D, forwards)
    check_link(result[1], [D], R, backwards)


def test_multilink_forwards():
    result = multi_link([R, D], [L, B], forwards)
    assert len(result) == 2
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_using(result[0], (3, 4), 9)
    check_using(result[1], (3, 4), 20)


def test_multilink_backwards():
    result = multi_link([R, D], [L, B], backwards=backwards)
    assert len(result) == 2
    check_link(result[0], [L, B], R)
    check_link(result[1], [L, B], D)
    check_using(result[0], (9, 20), 3)
    check_using(result[1], (9, 20), 4)


def test_multilink_forwards_backwards():
    result = multi_link([R, D], [L, B], forwards, backwards)
    assert len(result) == 4
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_link(result[2], [L, B], R)
    check_link(result[3], [L, B], D)
    check_using(result[0], (3, 4), 9)
    check_using(result[1], (3, 4), 20)
    check_using(result[2], (9, 20), 3)
    check_using(result[3], (9, 20), 4)


def test_multilink_nofunc():
    with pytest.raises(TypeError) as exc:
        multi_link([R, D], [L, B])
    assert exc.value.args[0] == "Must supply either forwards or backwards"


def test_linksame_string():
    """String inputs auto-converted to component IDs"""
    # ComponentLink does type checking to ensure conversion happens
    links = LinkSame('a', 'b')


def test_identity():
    assert lh.identity('3') == '3'


def test_toid():
    assert lh._toid('test').label == 'test'

    cid = ComponentID('test2')
    assert lh._toid(cid) is cid

    with pytest.raises(TypeError) as exc:
        lh._toid(None)


@pytest.mark.parametrize(('ndata', 'ndim'),
                         [(1, 1), (2, 0), (2, 1), (2, 2), (3, 2)])
def test_link_aligned(ndata, ndim):
    ds = []
    shp = tuple([2] * ndim)
    for i in range(ndata):
        d = Data()
        c = Component(np.random.random(shp))
        d.add_component(c, 'test')
        ds.append(d)

    # assert that all componentIDs are interchangeable
    links = LinkAligned(ds)
    dc = DataCollection(ds)
    dc.add_link(links)

    for i in range(ndim):
        id0 = ds[0].get_pixel_component_id(i)
        for j in range(1, ndata):
            id1 = ds[j].get_pixel_component_id(i)
            np.testing.assert_array_equal(ds[j][id0], ds[j][id1])
