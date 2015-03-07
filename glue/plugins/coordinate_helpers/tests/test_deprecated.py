import numpy as np

import pytest
from ....tests.helpers import ASTROPY_GE_04_INSTALLED

if not ASTROPY_GE_04_INSTALLED:
    pytest.skip()

from ....core.tests.test_link_helpers import check_link, check_using
from ....core import ComponentLink, ComponentID

from ..deprecated import Galactic2Equatorial, lb2ra, lb2dec, radec2glon, radec2glat, fk52gal, gal2fk5

R, D, L, B = (ComponentID('ra'), ComponentID('dec'),
              ComponentID('lon'), ComponentID('lat'))


def test_Galactic2Equatorial():
    result = Galactic2Equatorial(L, B, R, D)
    assert len(result) == 4
    check_link(result[0], [R, D], L)
    check_link(result[1], [R, D], B)
    check_link(result[2], [L, B], R)
    check_link(result[3], [L, B], D)
    x = np.array([45])
    y = np.array([50])    
    check_using(result[0], (x, y), 143.12136866)
    check_using(result[1], (x, y), -7.76422226)
    check_using(result[2], (x, y), 238.23062386)
    check_using(result[3], (x, y), 27.96352696)


def test_galactic2equatorial_individual():

    r = ComponentLink([L, B], R, lb2ra)
    d = ComponentLink([L, B], R, lb2dec)
    l = ComponentLink([R, D], L, radec2glon)
    b = ComponentLink([R, D], B, radec2glat)

    x = np.array([45])
    y = np.array([50])
    check_using(l, (x, y), 143.12136866)
    check_using(b, (x, y), -7.76422226)
    check_using(r, (x, y), 238.23062386)
    check_using(d, (x, y), 27.96352696)