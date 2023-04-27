# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u

from glue.core import ComponentID, Data, DataCollection
from glue.config import settings, unit_converter

from .. import link_helpers as lh
from ..link_helpers import (LinkTwoWay, MultiLink,
                            LinkSame, LinkAligned,
                            LinkSameWithUnits)


def setup_function(func):
    func.ORIGINAL_UNIT_CONVERTER = settings.UNIT_CONVERTER


def teardown_function(func):
    settings.UNIT_CONVERTER = func.ORIGINAL_UNIT_CONVERTER


def teardown_module():
    unit_converter._members.pop('test-spectral1')


R, D, L, B = (ComponentID('ra'), ComponentID('dec'),
              ComponentID('lon'), ComponentID('lat'))


def forwards(x):
    print('forwards input', x)
    return x * 3


def backwards(x):
    print('backwards input', x)
    return x / 3


def forwards_xy(x, y):
    print('forwards inputs', x, y)
    return x * 3, y * 5


def backwards_xy(x, y):
    print('backwards inputs', x, y)
    return x / 3, y / 5


def check_link(link, from_, to, using=None):
    assert link.get_from_ids() == from_
    assert link.get_to_id() == to
    if using:
        assert link.get_using() == using


def check_using(link, inp, out):
    assert_allclose(link.get_using()(*inp), out)


def test_LinkTwoWay():
    result = list(LinkTwoWay(R, D, forwards, backwards))
    check_link(result[0], [R], D, forwards)
    check_link(result[1], [D], R, backwards)


def test_multilink_forwards():
    result = MultiLink([R, D], [L, B], forwards_xy, labels2=['x', 'y'])
    assert len(result) == 2
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_using(result[0], (3, 4), 9)
    check_using(result[1], (3, 4), 20)


def test_multilink_backwards():
    result = MultiLink([R, D], [L, B], backwards=backwards_xy, labels1=['x', 'y'])
    assert len(result) == 2
    check_link(result[0], [L, B], R)
    check_link(result[1], [L, B], D)
    check_using(result[0], (9, 20), 3)
    check_using(result[1], (9, 20), 4)


def test_multilink_forwards_backwards():
    result = MultiLink([R, D], [L, B], forwards_xy, backwards_xy)
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
        MultiLink([R, D], [L, B])
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


@pytest.mark.parametrize('ndim', [0, 1, 2])
def test_link_aligned(ndim):
    shp = tuple([2] * ndim)
    data1 = Data(test=np.random.random(shp))
    data2 = Data(test=np.random.random(shp))

    links = LinkAligned(data1, data2)
    dc = DataCollection([data1, data2])
    dc.add_link(links)

    for i in range(ndim):
        id0 = data1.pixel_component_ids[i]
        id1 = data2.pixel_component_ids[i]
        assert_allclose(data1[id0], data2[id1])


def test_link_units():

    data1 = Data(weight_kg=[1, 2, 3])
    data2 = Data(weight_g=[2000, 3000, 1000])

    data1.get_component(data1.id['weight_kg']).units = 'kg'
    data2.get_component(data2.id['weight_g']).units = 'g'

    links = LinkSameWithUnits(data1.id['weight_kg'], data2.id['weight_g'])
    dc = DataCollection([data1, data2])
    dc.add_link(links)

    assert_allclose(data1[data1.id['weight_kg']], [1, 2, 3])
    assert_allclose(data2[data1.id['weight_kg']], [2, 3, 1])
    assert_allclose(data1[data2.id['weight_g']], [1000, 2000, 3000])
    assert_allclose(data2[data2.id['weight_g']], [2000, 3000, 1000])


def test_link_units_invalid():

    data1 = Data(weight_kg=[1, 2, 3])
    data2 = Data(distance_m=[2000, 3000, 1000])

    data1.get_component(data1.id['weight_kg']).units = 'kg'
    data2.get_component(data2.id['distance_m']).units = 'm'

    with pytest.raises(u.UnitConversionError, match='not convertible'):
        LinkSameWithUnits(data1.id['weight_kg'], data2.id['distance_m'])


@unit_converter('test-spectral1')
class SpectralUnitConverter:

    def equivalent_units(self, data, cid, units):
        return map(str, u.Unit(units).find_equivalent_units(include_prefix_units=True, equivalencies=u.spectral()))

    def to_unit(self, data, cid, values, original_units, target_units):
        return (values * u.Unit(original_units)).to_value(target_units, equivalencies=u.spectral())


def test_link_units_custom():

    data1 = Data(wav_mm=[1, 2, 3])
    data2 = Data(freq_ghz=[10, 20, 30])

    data1.get_component(data1.id['wav_mm']).units = 'mm'
    data2.get_component(data2.id['freq_ghz']).units = 'GHz'

    with pytest.raises(u.UnitConversionError, match='not convertible'):
        LinkSameWithUnits(data1.id['wav_mm'], data2.id['freq_ghz'])

    settings.UNIT_CONVERTER = 'test-spectral1'

    links = LinkSameWithUnits(data1.id['wav_mm'], data2.id['freq_ghz'])
    dc = DataCollection([data1, data2])
    dc.add_link(links)

    assert_allclose(data1[data1.id['wav_mm']], [1, 2, 3])
    assert_allclose(data2[data1.id['wav_mm']], [29.9792, 14.9896, 9.9930], rtol=1e-3)
    assert_allclose(data1[data2.id['freq_ghz']], [299.7924, 149.8962, 99.9308], rtol=1e-3)
    assert_allclose(data2[data2.id['freq_ghz']], [10, 20, 30])
