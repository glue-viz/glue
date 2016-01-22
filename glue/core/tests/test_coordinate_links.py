from __future__ import absolute_import, division, print_function

import numpy as np

from glue.tests.helpers import requires_astropy, ASTROPY_INSTALLED

from .. import Data, DataCollection
from ..coordinates import coordinates_from_header
from ..link_helpers import LinkSame
from glue.tests.helpers import make_file

if ASTROPY_INSTALLED:
    from astropy.io import fits


@requires_astropy
def test_wcs_3d_to_2d():
    """ For a "normal" XYV cube, linking XY world should be
    enough to propagate XY pixel
    """
    d = Data(label='D1')
    with make_file(test_fits, suffix='.fits', decompress=True) as file:
        header = fits.getheader(file)
    d.coords = coordinates_from_header(header)
    d.add_component(np.zeros((3, 2, 1)), label='test')

    d2 = Data(label='D2')
    d2.coords = coordinates_from_header(header)
    d2.add_component(np.zeros((3, 2, 1)), label='test2')

    dc = DataCollection([d, d2])
    dc.add_link(LinkSame(d.get_world_component_id(1),
                         d2.get_world_component_id(1)))
    dc.add_link(LinkSame(d.get_world_component_id(2),
                         d2.get_world_component_id(2)))

    py = d.get_pixel_component_id(1)
    px = d.get_pixel_component_id(2)
    py2 = d2.get_pixel_component_id(1)
    px2 = d2.get_pixel_component_id(2)

    np.testing.assert_array_almost_equal(d2[px], d2[px2])
    np.testing.assert_array_almost_equal(d2[py], d2[py2])


@requires_astropy
def test_link_velocity():
    """ For a normal PPV cube, linking velocity world should be
    enough to get pixel V"""
    d = Data(label='D1')
    with make_file(test_fits, suffix='.fits', decompress=True) as file:
        header = fits.getheader(file)
    d.coords = coordinates_from_header(header)
    d.add_component(np.zeros((3, 2, 1)), label='test')

    d2 = Data(label='D2')
    d2.coords = coordinates_from_header(header)
    d2.add_component(np.zeros((3, 2, 1)), label='test2')

    dc = DataCollection([d, d2])
    dc.add_link(LinkSame(d.get_world_component_id(0),
                         d2.get_world_component_id(0)))

    pz = d.get_pixel_component_id(0)
    pz2 = d2.get_pixel_component_id(0)

    np.testing.assert_array_almost_equal(d2[pz], d2[pz2])


@requires_astropy
class TestDependentAxes(object):

    def test_base(self):
        d = Data(x=[1, 2, 3])
        assert d.coords.dependent_axes(0) == (0,)

        d = Data(x=[[1, 2], [3, 4]])
        assert d.coords.dependent_axes(0) == (0,)
        assert d.coords.dependent_axes(1) == (1,)

    def header2(self, proj='SIN'):
        result = fits.Header()
        result['NAXIS'] = 2
        result['NAXIS1'] = 100
        result['NAXIS2'] = 100
        result['CRPIX1'] = 1
        result['CRPIX2'] = 1
        result['CDELT1'] = 1
        result['CDELT2'] = 1
        result['CTYPE1'] = 'RA---%s' % proj
        result['CTYPE2'] = 'DEC--%s' % proj
        result['CRVAL1'] = 1
        result['CRVAL2'] = 1
        return result

    def header3(self, proj='SIN'):
        result = self.header2(proj)
        result.update(NAXIS=3, NAXIS3=1, CDELT3=1,
                      CRPIX3=3, CTYPE3='VOPT')
        return result

    def header4(self):
        result = fits.Header()
        result.update(WCSAXES=4,
                      CRPIX1=513,
                      CRPIX2=513,
                      CRPIX3=1,
                      CRPIX4=1,
                      CDELT1=-6.94444444444E-05,
                      CDELT2=6.94444444444E-05,
                      CDELT3=10000.1667626,
                      CDELT4=1,
                      CTYPE1='RA---SIN',
                      CTYPE2='DEC--SIN',
                      CTYPE3='VOPT',
                      CTYPE4='STOKES',
                      CRVAL1=56.7021416715,
                      CRVAL2=68.0961055596,
                      CRVAL3=-280000.000241,
                      CRVAL4=1,
                      PV2_1=0,
                      PV2_2=0,
                      LONPOLE=180,
                      LATPOLE=68.0961055596,
                      RESTFRQ=34596380000,
                      RADESYS='FK5',
                      EQUINOX=2000,
                      SPECSYS='BARYCENT')
        return result

    def test_wcs_ppv(self):

        header = self.header3()

        d = Data(label='D1')
        d.coords = coordinates_from_header(header)
        d.add_component(np.zeros((3, 2, 1)), label='test')

        assert d.coords.dependent_axes(0) == (0,)
        assert d.coords.dependent_axes(1) == (1, 2)
        assert d.coords.dependent_axes(2) == (1, 2)

    def test_wcs_alma(self):
        header = self.header4()

        d = Data(label='D1')
        d.coords = coordinates_from_header(header)
        d.add_component(np.zeros((3, 2, 1, 1)), label='test')

        assert d.coords.dependent_axes(0) == (0,)
        assert d.coords.dependent_axes(1) == (1,)
        assert d.coords.dependent_axes(2) == (2, 3)
        assert d.coords.dependent_axes(3) == (2, 3)


test_fits = b'x\x9c\xed\x97Qs\xa2H\x14\x85\xf7\xa7\xdc\xa75I\x05B\x83\xa0\xb8\x95\x07\xd462\x11q\xa0\xcdL\xe6%\x85\xd21T!X\x80\x93\xf1\xdf\xef\x05uuv\xccN\xc0<mq\x1e\x04\x11>O\x9f\xee\xa6o\xbb\xa65\x19Q\x80[8!\x0670\x8f\xa3\xe78Y\xa6\x90\xc500\x99\x0bi\xe6E\xbe\x97\xf8\xa7\x1e\x00\xe8\x9alb~=\xc9\x13\xb4&\xf2\xbc$\xf16\xe0{\x99\x07\xd9f\xc5OS\x0e\x1a\x1b_M\x17\xde\xf0\xa7 /Z/g<\x81\xf8yO\x0e\x96<J\x838J\xdf\xe6\x917x\xe4wn\xde\xe0\xc9\x1f\xccS>\x8e\xd7\xb3-\x8b\x8e\x19\x9e\x15\x9dw1\x08\xf9\x8f`\x16r0\x97\xde\x82\x03K\xbc(]\xc5I\x06\xee&\xcd\xf8\xf2\x12\xf2\xce\xf62\x08R\xf0\xf9s\x10q\x1f\x82\x08\x1aF\x9a%q\x14/7\x07\x1e\x8e\x02(.\xaf^6i0O\x1b\xd7\xf0=\x0e\xd7K\x0eJK\xbb\x86U\x8eWT\xfd/\x98\x05\xb3y\xec\xf3\x0e\xc8\x92D\x8c?\rQ\x14\xf1\x0e\xfcP\xf5!\xf4\rFs\x9f\xb7\xd0\xc0\x9f\x9b\x82\xa4\x08De2\xe9\xc8\xed\x8e,7\xb0\x83\x9f\x03t;O\xb8\x97a\xa7\xe6\x03\x87\xc3\xc5#J\xb0,\xa1\xdfg//\x9d\xe5\xb2\x93\xa60e\x97\xc8\xb1\xbb\x9fh\x8f\x15\xbc\tu\\:u\x8b\x18\x1a\xbb8n\xca\xe6\xc7\xe8\x88\xba={\x82\xbcA\xcf1l\x814\xad\xc6\xe1\xe7\xd2<s\xec2gjQ\xe4\xb9\xf4\xf3\xd46\r8\xc2\x95\xe7\xd9]\xf7)\xcfp7^0CY\x94u\xa2\xabDk\x11R\x9e7e\xdb\xe3\xcf\xe3\x8f(bS\xd7\x8a\xf9;\xb4\xa7\x8e\xfb~\xde\xc8e\'x\xb2\x8cC@\xd1\x95\xf2<:\xb1{\xc3_xP\xb4\\\x12\xcb\xb7\xd7\xf8fZS6\xdc\xf1\xb4\xb6\xd8&\x8a\xdeV\xe5\x96Nd\xb9\x8d\xbc>\xbds(}\xb7C\x1c.\x0f\x063\xed-O\xd6DM\xd1\x9b\xedv\x13\x91\x9a\xa2\x95\xe7=\x8c\\\'?\x9ex\x1f\x90\xa2\xbd\xf7\xd6M\x89\xf8\xa0;\x1d\x9b\xac\xe0\xe5\xc3\xee>\xbf\xf4\xf3\xf8c\xc6U\t\x1c\xb8\xf7\x8fO\x03\x87~\xde\xfa#\xe8\x89\xb4u\x89\xb4U]\x95\x9b*\xf2\xee\x86\xdf\xca\xf0F\xe6\x98\x1ex;\xe5XY"J\xab<\xaf\xfbe{<\x91\x9f\xac\xe6\xf9Y\xe5\xfd\x8d\x8db\xfe\x12\xa5g?\x11A:k\xfe\xf6\xd8\xe3\x84\x16\xebQ\xc31\x04Ap\x07\xa3\xf3x\xce\x831:^\xdf\xd4\x96\xa8\xa9\xfaN\x15x}:b{\x9e \x89\x92\xa4)m\x94.\xe5\xaa\xe2\x0f\xcb\x83c\x7fJ\xce\xdc\xabb~\xc5\xfa\xdb\xe8\xd3\xde\x07\xe5\xf7\xebz\xbe3Y1\xbf\x7fx\x1f\x94\xdf\x91?\xa1\xa9\xe9MQ\'\xca9\xf9\x15\xf5F\xe3\x81\x8el\x01_7\xe7\xe7wT\xbf\x08\xba\xaea\xab[\xba$\xb7Z\xad\x8a\xf9\x1d\xd7C\x9a&6eE#\xe7\xe4w\xaa\xbeB\xa3\xa2T\x96\x0604]f;\x8fp\xc7#\x9e`m\xe2\xc3l\x03E\xa5\x00\x0e_$\x81\xef\x07\xd1\x02&I\xbcH\xbc%\xe0\xda\xfc\x9b\xff\xd8\xf3\xba^\xcaC,\xbf\xc0]\xcf\xb2\xc4\x9b\xe7\xe4*\xda\xf3\n\xdd\x85\xf1\xcc\x0b\x0f\xec\x89\x87\xa6x\xc6\x93\xb4\x83\xb5e\x9c\xf8XH\xaf\xe2p\x83\x85^\x80\xf7\xfd\x17\xefK\x10\xf9\xf1+0,\xe1o\xbf\xf30\x9e\x07\xd9\xe6l\x7fE}\x9b\xcf\x11\xc8\xd7\xf3\xed\xb1"o\x1c\x07)\x87W\x1e,^\xb2\xbc\x07\xc8\xcd*~\xbdp\xcd;\xcb\xb8\x96/\xab\xf0`\x10b]<\x08xXt)n8\xfa\xf9&\xa6\xa2?w\x85\xf5,f<\x08B\xcc\xbf\x03\x9f\x82h~\xb5\xf0\xd6i\x1ax\xd1U\xfe\xad\x1c\xcf\x8cV\xeb\x0cl6\x00w\x8e%}\xa7\xac\xaf\x7f\xf3@ST-\xdf\xf3\xe1I\xab\xc2\xbec/:\xeeW\x7f\xb8V\xadZ\xb5j\xd5\xaa\xf5\xbf\xd4\x1f\xb5j\xd5\xaaU\xabV\xadZ\xb5j\xd5z\xb7\xfe\x06\xb6\x02\x94\xfe'
