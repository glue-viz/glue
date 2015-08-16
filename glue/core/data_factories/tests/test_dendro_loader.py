from __future__ import print_function, division

import os

import pytest
import numpy as np

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


@requires_astrodendro
def test_identifier_heuristics(tmpdir):

    filename = tmpdir.join('test.fits').strpath

    from ..dendro_loader import is_dendro

    from ....external.astro import fits

    hdulist = fits.HDUList()
    hdulist.append(fits.PrimaryHDU())
    hdulist.append(fits.ImageHDU())
    hdulist.append(fits.ImageHDU())

    hdulist.writeto(filename)
    assert not is_dendro(filename)

    hdulist.append(fits.ImageHDU())

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[1].name = 'random'

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[1].name = ''
    hdulist[0].data = np.array([1,2,3])

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[0].data = None
    hdulist[1].data = np.ones((3,4))
    hdulist[2].data = np.ones((2,4))
    hdulist[3].data = np.ones((3,5))

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[2].data = np.ones((3,4))

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[3].data = np.ones(3)

    hdulist.writeto(filename, clobber=True)
    assert is_dendro(filename)
