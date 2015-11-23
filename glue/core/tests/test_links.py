"""This file contains tests concerning linking data and accessing
linked components"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.random import random as r

from .. import Data, DataCollection
from ..link_helpers import LinkSame


def test_1d_world_link():
    x, y = r(10), r(10)
    d1 = Data(label='d1', x=x)
    d2 = Data(label='d2', y=y)
    dc = DataCollection([d1, d2])

    dc.add_link(LinkSame(d2.get_world_component_id(0), d1.id['x']))

    assert d2.get_world_component_id(0) in d1.components
    np.testing.assert_array_equal(d1[d2.get_world_component_id(0)], x)
    np.testing.assert_array_equal(d1[d2.get_pixel_component_id(0)], x)


def test_3d_world_link():
    """Should be able to grab pixel coords after linking world"""
    x, y, z = r(10), r(10), r(10)
    cat = Data(label='cat', x=x, y=y, z=z)
    im = Data(label='im', inten=r((3, 3, 3)))

    dc = DataCollection([cat, im])

    dc.add_link(LinkSame(im.get_world_component_id(2), cat.id['x']))
    dc.add_link(LinkSame(im.get_world_component_id(1), cat.id['y']))
    dc.add_link(LinkSame(im.get_world_component_id(0), cat.id['z']))

    np.testing.assert_array_equal(cat[im.get_pixel_component_id(2)], x)
    np.testing.assert_array_equal(cat[im.get_pixel_component_id(1)], y)
    np.testing.assert_array_equal(cat[im.get_pixel_component_id(0)], z)


def test_2d_world_link():
    """Should be able to grab pixel coords after linking world"""

    x, y = r(10), r(10)
    cat = Data(label='cat', x=x, y=y)
    im = Data(label='im', inten=r((3, 3)))

    dc = DataCollection([cat, im])

    dc.add_link(LinkSame(im.get_world_component_id(0), cat.id['x']))
    dc.add_link(LinkSame(im.get_world_component_id(1), cat.id['y']))

    np.testing.assert_array_equal(cat[im.get_pixel_component_id(0)], x)
    np.testing.assert_array_equal(cat[im.get_pixel_component_id(1)], y)
