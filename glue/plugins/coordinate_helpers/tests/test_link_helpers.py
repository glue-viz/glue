import numpy as np

import pytest
from ....tests.helpers import ASTROPY_GE_04_INSTALLED

if not ASTROPY_GE_04_INSTALLED:
    pytest.skip()

from ..link_helpers import Galactic2Equatorial, lb2ra, lb2dec, radec2glon, radec2glat, fk52gal, gal2fk5
from ....core.tests.test_link_helpers import check_link, check_using
from ....core import ComponentLink, ComponentID

R, D, L, B = (ComponentID('ra'), ComponentID('dec'),
              ComponentID('lon'), ComponentID('lat'))


def test_Galactic2Equatorial():
    result = Galactic2Equatorial(L, B, R, D)
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


def test_galactic2equatorial_individual():

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