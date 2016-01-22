from __future__ import absolute_import, division, print_function

import os

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from glue.tests.helpers import make_file

from glue.core.data_factories.helpers import find_factory
from glue.core import data_factories as df
from glue.tests.helpers import requires_astrodendro


DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_astrodendro
@pytest.mark.parametrize('filename', ['dendro.fits', 'dendro_old.fits', 'dendro.hdf5'])
def test_is_dendro(filename):
    from ..data_factory import is_dendro
    assert is_dendro(os.path.join(DATA, filename))


@requires_astrodendro
@pytest.mark.parametrize('filename', ['dendro.fits', 'dendro_old.fits', 'dendro.hdf5'])
def test_find_factory(filename):
    from ..data_factory import load_dendro
    assert find_factory(os.path.join(DATA, filename)) is load_dendro


@requires_astrodendro
def test_identifier_heuristics(tmpdir):

    filename = tmpdir.join('test.fits').strpath

    from ..data_factory import is_dendro

    from astropy.io import fits

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
    hdulist[0].data = np.array([1, 2, 3])

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[0].data = None
    hdulist[1].data = np.ones((3, 4))
    hdulist[2].data = np.ones((2, 4))
    hdulist[3].data = np.ones((3, 5))

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[2].data = np.ones((3, 4))

    hdulist.writeto(filename, clobber=True)
    assert not is_dendro(filename)

    hdulist[3].data = np.ones(3)

    hdulist.writeto(filename, clobber=True)
    assert is_dendro(filename)


@requires_astrodendro
def test_dendrogram_load():
    from ..data_factory import load_dendro
    data = b"""x\xda\xed\xda]K\xc2`\x18\xc6\xf1^\xbe\xc8}fA\xe4[X\x14\x1eX\x99<\x90S\xd8\x02O\x9f\xf2Q<\xd8&\xcf&\xe4\xb7\xcft\x82\xc9\xe6\x1be\x91\xff\xdf\xc9\xc5\xd8v\xc1vt\xeff\xaej\xb6\x9f\xeb"UI\xe1I^\xde\xc2\xa0\x17Z?\x928\x94\'\xe5\xb9\x12\xc5:\xe8j\xdb\x95T\xf7\xcak\xabNF\xdf\xcd\xa4O[\xab\xc7\xd2\xd5\xb1\x96x<4\xb2\x86S\xeb(W2\xfa\n\x93\xbe`\xe4\xbf\x1a+ao\xde<\xf0M\x10\r\xc2 J\xed\xabw\xbc\xba\xf3\x98\xf9\xbc[\x9b\x96\x01\x00\x00\xe0`|\x8e\x93\xaej9U\xc9\xa9f\xad1\x99\xa4%\xb7p:/\xca\xd7}#\xe6=\x9eM\xa5\xeb\xfaV\xcd\xcf\x95\xabo\x9e\x9f\x8b\xdb\xcf\xcf\xd3\xbebF_e\xfb\xf7\xd7~h\xbd8\xdeF\xf3\xfdP[\xed\x9b\xd8\xd8hE_cU\xdf\xd7\xe7\xed\xdbp4\x8c\x98\xef\x01\x00\x00\xf6\xeah\xe68\xc9\x93$O3\x8e\xe7\xd7\x01\x00\x00\x00\x07i\x9f\xfb\xe7r\x89\xfd3\xfbg\x00\x00\x80\x7f\xb1\x7fN\xdbA\x03\x00\x00\x00\xf8\xc5\xfd\xf3_\xff\xff\xb9t\xcd\xfe\x19\x00\x00\x00\x1b\xed\x9f\xcf\x96\xb2\x98\xe4m\x92\xe5$/\x93,d\xe4E\x92\xa5\x1d\xef?_:\xde\xf5\xfe;\xbe\x8c\x00\x00\x00\xf0\x13>\x00\x8e\xbe x"""
    with make_file(data, 'fits', decompress=True) as fname:
        dg, im = df.load_data(fname, factory=load_dendro)
    assert_array_equal(im['intensity'], [1, 2, 3, 2, 3, 1])
    assert_array_equal(im['structure'], [0, 0, 1, 0, 2, 0])
    assert_array_equal(dg['parent'], [-1, 0, 0])
    assert_array_equal(dg['height'], [3, 3, 3])
    assert_array_equal(dg['peak'], [3, 3, 3])
