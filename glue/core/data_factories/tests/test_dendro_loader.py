from __future__ import print_function, division

import os

import pytest

from ..helpers import find_factory
from ....tests.helpers import requires_astrodendro



DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_astrodendro
@pytest.mark.parametrize('filename', ['dendro.fits', 'dendro_old.fits', 'dendro.hdf5'])
def test_is_dendro(filename):
    from ..dendro_loader import is_dendro
    assert is_dendro(os.path.join(DATA, filename))


@requires_astrodendro
@pytest.mark.parametrize('filename', ['dendro.fits', 'dendro_old.fits', 'dendro.hdf5'])
def test_find_factory(filename):
    from ..dendro_loader import load_dendro
    assert find_factory(os.path.join(DATA, filename)) is load_dendro