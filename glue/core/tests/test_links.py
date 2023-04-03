"""This file contains tests concerning linking data and accessing
linked components"""

import numpy as np
from numpy.random import random as r

from glue.core.coordinates import IdentityCoordinates

from .. import Data, DataCollection
from ..link_helpers import LinkSame


def test_1d_world_link():
    x, y = r(10), r(10)
    d1 = Data(label='d1', x=x)
    d2 = Data(label='d2', y=y, coords=IdentityCoordinates(n_dim=1))
    dc = DataCollection([d1, d2])

    dc.add_link(LinkSame(d2.world_component_ids[0], d1.id['x']))

    assert d2.world_component_ids[0] in d1.externally_derivable_components

    np.testing.assert_array_equal(d1[d2.world_component_ids[0]], x)
    np.testing.assert_array_equal(d1[d2.pixel_component_ids[0]], x)


def test_3d_world_link():
    """Should be able to grab pixel coords after linking world"""
    x, y, z = r(10), r(10), r(10)
    cat = Data(label='cat', x=x, y=y, z=z)
    im = Data(label='im', inten=r((3, 3, 3)), coords=IdentityCoordinates(n_dim=3))

    dc = DataCollection([cat, im])

    dc.add_link(LinkSame(im.world_component_ids[2], cat.id['x']))
    dc.add_link(LinkSame(im.world_component_ids[1], cat.id['y']))
    dc.add_link(LinkSame(im.world_component_ids[0], cat.id['z']))

    np.testing.assert_array_equal(cat[im.pixel_component_ids[2]], x)
    np.testing.assert_array_equal(cat[im.pixel_component_ids[1]], y)
    np.testing.assert_array_equal(cat[im.pixel_component_ids[0]], z)


def test_2d_world_link():
    """Should be able to grab pixel coords after linking world"""

    x, y = r(10), r(10)
    cat = Data(label='cat', x=x, y=y)
    im = Data(label='im', inten=r((3, 3)), coords=IdentityCoordinates(n_dim=2))

    dc = DataCollection([cat, im])

    dc.add_link(LinkSame(im.world_component_ids[0], cat.id['x']))
    dc.add_link(LinkSame(im.world_component_ids[1], cat.id['y']))

    np.testing.assert_array_equal(cat[im.pixel_component_ids[0]], x)
    np.testing.assert_array_equal(cat[im.pixel_component_ids[1]], y)
