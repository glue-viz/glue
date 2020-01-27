import pytest
import numpy as np
from glue.core import Data
from glue.core.coordinates import WCSCoordinates
from astropy.io import fits

from ..gridded_fits import fits_writer

BITPIX = {}
BITPIX[np.int16] = 16
BITPIX[np.int32] = 32
BITPIX[np.int64] = 64
BITPIX[np.float32] = -32
BITPIX[np.float64] = -64


@pytest.mark.parametrize('dtype', BITPIX.keys())
def test_fits_writer_data(tmpdir, dtype):

    dtype = np.dtype(dtype)

    filename = tmpdir.join('test1.fits').strpath

    data = Data(x=np.arange(6).reshape(2, 3).astype(dtype),
                y=(np.arange(6) * 2).reshape(2, 3).astype(dtype))

    fits_writer(filename, data)

    with fits.open(filename) as hdulist:
        assert len(hdulist) == 2
        np.testing.assert_equal(hdulist['x'].data, data['x'])
        np.testing.assert_equal(hdulist['y'].data, data['y'])
        # Note: the following tolerates endian-ness change
        assert hdulist['x'].data.dtype in (dtype, dtype.newbyteorder())
        assert hdulist['y'].data.dtype in (dtype, dtype.newbyteorder())

    # Only write out some components

    filename = tmpdir.join('test2.fits').strpath

    fits_writer(filename, data, components=[data.id['x']])

    with fits.open(filename) as hdulist:
        assert len(hdulist) == 1
        np.testing.assert_equal(hdulist['x'].data, data['x'])


def test_component_unit_header(tmpdir):
    from astropy import units as u
    filename = tmpdir.join('test3.fits').strpath

    data = Data(x=np.arange(6).reshape(2, 3),
                y=(np.arange(6) * 2).reshape(2, 3),
                z=(np.arange(6) * 2).reshape(2, 3))

    data.coords = WCSCoordinates()

    unit1 = data.get_component("x").units = u.m / u.s
    unit2 = data.get_component("y").units = u.Jy
    unit3 = data.get_component("z").units = ""

    fits_writer(filename, data)

    with fits.open(filename) as hdulist:
        assert len(hdulist) == 3
        bunit = hdulist['x'].header.get('BUNIT')
        assert u.Unit(bunit) == unit1

        bunit = hdulist['y'].header.get('BUNIT')
        assert u.Unit(bunit) == unit2

        bunit = hdulist['z'].header.get('BUNIT')
        assert bunit == unit3


@pytest.mark.parametrize('dtype', BITPIX.keys())
def test_fits_writer_subset(tmpdir, dtype):

    filename = tmpdir.join('test').strpath

    data = Data(x=np.arange(6).reshape(2, 3).astype(dtype),
                y=(np.arange(6) * 2).reshape(2, 3).astype(dtype))

    subset = data.new_subset()
    subset.subset_state = data.id['x'] > 2

    fits_writer(filename, subset)

    with fits.open(filename) as hdulist:
        assert np.all(np.isnan(hdulist['x'].data[0]))
        assert np.all(np.isnan(hdulist['y'].data[0]))
        np.testing.assert_equal(hdulist['x'].data[1], data['x'][1])
        np.testing.assert_equal(hdulist['y'].data[1], data['y'][1])
        # Here we check BITPIX, not the dtype of the read in data, because if
        # BLANK is present, astropy.io.fits scales the data to float. We want to
        # just make sure here the data is stored with the correct type on disk.
        assert hdulist['x'].header['BITPIX'] == BITPIX[dtype]
        assert hdulist['y'].header['BITPIX'] == BITPIX[dtype]
