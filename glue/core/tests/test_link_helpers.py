#pylint: disable=W0613,W0201,W0212,E1101,E1103
import pytest
import numpy as np

from ..link_helpers import *
from ...core import ComponentID, ComponentLink

R, D, L, B = (ComponentID('ra'), ComponentID('dec'),
              ComponentID('lon'), ComponentID('lat'))


def forwards(x, y):
    print 'forwads inputs', x, y
    return x * 3, y * 5


def backwards(x, y):
    print 'backwards inputs', x, y
    return x / 3, y / 5


def check_link(link, from_, to, using=None):
    assert link.get_from_ids() == from_
    assert link.get_to_id() == to
    if using:
        assert link.get_using() == using


def check_using(link, inp, out):
    np.testing.assert_array_almost_equal(link.get_using()(*inp), out)


def test_link_twoway():
    result = link_twoway(R, D, forwards, backwards)
    check_link(result[0], [R], D, forwards)
    check_link(result[1], [D], R, backwards)


def test_multilink_forwards():
    result = multilink([R, D], [L, B], forwards)
    assert len(result) == 2
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_using(result[0], (3, 4), 9)
    check_using(result[1], (3, 4), 20)


def test_multilink_backwards():
    result = multilink([R, D], [L, B], backwards=backwards)
    assert len(result) == 2
    check_link(result[0], [L, B], R)
    check_link(result[1], [L, B], D)
    check_using(result[0], (9, 20), 3)
    check_using(result[1], (9, 20), 4)


def test_multilink_forwards_backwards():
    result = multilink([R, D], [L, B], forwards, backwards)
    assert len(result) == 4
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_link(result[2], [L, B], R)
    check_link(result[3], [L, B], D)
    check_using(result[0], (3, 4), 9)
    check_using(result[1], (3, 4), 20)
    check_using(result[2], (9, 20), 3)
    check_using(result[3], (9, 20), 4)


def test_galactic2ecliptic():
    from aplpy.wcs_util import fk52gal, gal2fk5
    result = galactic2ecliptic(L, B, R, D)
    assert len(result) == 4
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_link(result[2], [L, B], R)
    check_link(result[3], [L, B], D)
    x = np.array([0])
    y = np.array([0])
    check_using(result[0], (x, y), fk52gal(x, y)[0])
    check_using(result[1], (x, y), fk52gal(x, y)[1])
    check_using(result[2], (x, y), gal2fk5(x, y)[0])
    check_using(result[3], (x, y), gal2fk5(x, y)[1])


def test_galactic2ecliptic_individual():
    from aplpy.wcs_util import fk52gal, gal2fk5

    r = ComponentLink([L, B], R, lb2ra)
    d = ComponentLink([L, B], R, lb2dec)
    l = ComponentLink([R, D], L, radec2glon)
    b = ComponentLink([R, D], B, radec2glat)

    x = np.array([0])
    y = np.array([0])
    check_using(l, (x, y), fk52gal(x, y)[0])
    check_using(b, (x, y), fk52gal(x, y)[1])
    check_using(r, (x, y), gal2fk5(x, y)[0])
    check_using(d, (x, y), gal2fk5(x, y)[1])


def test_multilink_nofunc():
    with pytest.raises(TypeError) as exc:
        result = multilink([R, D], [L, B])
    assert exc.value.args[0] == "Must supply either forwards or backwards"
