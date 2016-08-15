from __future__ import absolute_import, division, print_function

import os
from glue.qglue import parse_data
from glue.tests.helpers import requires_spectral_cube

DATA = os.path.join(os.path.dirname(__file__), 'data')


@requires_spectral_cube
def test_identifier():
    from ..spectral_cube import is_spectral_cube
    assert is_spectral_cube(os.path.join(DATA, 'cube_3d.fits'))


@requires_spectral_cube
def test_reader():
    from ..spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(DATA, 'cube_3d.fits'))
    data['STOKES I']
    assert data.shape == (2, 3, 4)


@requires_spectral_cube
def test_qglue():
    from spectral_cube import SpectralCube
    cube = SpectralCube.read(os.path.join(DATA, 'cube_3d.fits'))
    data = parse_data(cube, 'x')[0]
    assert data.label == 'x'
    data['STOKES I']
    assert data.shape == (2, 3, 4)
