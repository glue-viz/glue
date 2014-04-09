import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits

from ..pvextractor import extract_pv_slice
from ..geometry.path import Path

# Use a similar header as in the spectral_cube package
HEADER_STR = """
SIMPLE  =                    T / Written by IDL:  Fri Feb 20 13:46:36 2009
BITPIX  =                  -32  /
NAXIS   =                    4  /
NAXIS1  =                    3  /
NAXIS2  =                    4  /
NAXIS3  =                    5  /
NAXIS4  =                    1  /
EXTEND  =                    T  /
BSCALE  =    1.00000000000E+00  /
BZERO   =    0.00000000000E+00  /
BLANK   =                   -1  /
TELESCOP= 'VLA     '  /
CDELT1  =   -5.55555561268E-04  /
CRPIX1  =    1.37300000000E+03  /
CRVAL1  =    2.31837500515E+01  /
CTYPE1  = 'RA---SIN'  /
CDELT2  =    5.55555561268E-04  /
CRPIX2  =    1.15200000000E+03  /
CRVAL2  =    3.05765277962E+01  /
CTYPE2  = 'DEC--SIN'  /
CDELT3  =    1.28821496879E+03  /
CRPIX3  =    1.00000000000E+00  /
CRVAL3  =   -3.21214698632E+05  /
CTYPE3  = 'VELO-HEL'  /
CDELT4  =    1.00000000000E+00  /
CRPIX4  =    1.00000000000E+00  /
CRVAL4  =    1.00000000000E+00  /
CTYPE4  = 'STOKES  '  /
DATE-OBS= '1998-06-18T16:30:25.4'  /
RESTFREQ=    1.42040571841E+09  /
CELLSCAL= 'CONSTANT'  /
BUNIT   = 'JY/BEAM '  /
EPOCH   =    2.00000000000E+03  /
OBJECT  = 'M33     '           /
OBSERVER= 'AT206   '  /
VOBS    =   -2.57256763070E+01  /
LTYPE   = 'channel '  /
LSTART  =    2.15000000000E+02  /
LWIDTH  =    1.00000000000E+00  /
LSTEP   =    1.00000000000E+00  /
BTYPE   = 'intensity'  /
DATAMIN =   -6.57081836835E-03  /
DATAMAX =    1.52362231165E-02  /"""


def make_test_hdu():
    header = fits.header.Header.fromstring(HEADER_STR, sep='\n')
    hdu = fits.PrimaryHDU(header=header)
    import numpy as np
    hdu.data = np.zeros((1, 5, 4, 3))
    hdu.data[:, :, 0, :] = 1.
    hdu.data[:, :, 2, :] = 2.
    hdu.data[:, :, 3, :] = np.nan
    return hdu


def test_pv_slice_hdu_line_path_order_0():
    hdu = make_test_hdu()
    path = Path([(1., -0.5), (1., 3.5)])
    slice_hdu = extract_pv_slice(hdu, path, spacing=0.4, order=0)
    assert_allclose(slice_hdu.data[0], np.array([1., 1., 0., 0., 0., 2., 2., 2., np.nan, np.nan]))


def test_pv_slice_hdu_line_path_order_3():
    hdu = make_test_hdu()
    path = Path([(1., -0.5), (1., 3.5)])
    slice_hdu = extract_pv_slice(hdu, path, spacing=0.4, order=3)
    assert_allclose(slice_hdu.data[0], np.array([np.nan, 0.9648, 0.4, -0.0368, 0.5622,
                                                 1.6478, 1.9278, np.nan, np.nan, np.nan]))


def test_pv_slice_hdu_poly_path():
    hdu = make_test_hdu()
    path = Path([(1., -0.5), (1., 3.5)], width=0.001)
    slice_hdu = extract_pv_slice(hdu, path, spacing=0.4)
    assert_allclose(slice_hdu.data[0], np.array([1., 1., 1., 0., 0., 1., 2., 2., np.nan, np.nan]))


def test_pv_slice_hdu_line_path_order_0_no_nan():
    hdu = make_test_hdu()
    path = Path([(1., -0.5), (1., 3.5)])
    slice_hdu = extract_pv_slice(hdu, path, spacing=0.4, order=0, respect_nan=False)
    assert_allclose(slice_hdu.data[0], np.array([1., 1., 0., 0., 0., 2., 2., 2., 0., 0.]))


def test_pv_slice_hdu_line_path_order_3_no_nan():
    hdu = make_test_hdu()
    path = Path([(1., -0.5), (1., 3.5)])
    slice_hdu = extract_pv_slice(hdu, path, spacing=0.4, order=3, respect_nan=False)
    assert_allclose(slice_hdu.data[0], np.array([np.nan, 0.9648, 0.4, -0.0368, 0.5622,
                                                 1.6478, 1.9278, 0.975, 0.0542, np.nan]))


def test_pv_slice_hdu_poly_path_no_nan():
    hdu = make_test_hdu()
    path = Path([(1., -0.5), (1., 3.5)], width=0.001)
    slice_hdu = extract_pv_slice(hdu, path, spacing=0.4, respect_nan=False)
    assert_allclose(slice_hdu.data[0], np.array([1., 1., 1., 0., 0., 1., 2., 2., 0., 0.]))
