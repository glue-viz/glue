from collections import OrderedDict
import pytest

from numpy.testing import assert_equal

from astropy.io import fits
from glue.io.formats.fits.subset_mask import fits_subset_mask_importer, fits_subset_mask_exporter


def test_reader(tmpdir):

    mask_filename = tmpdir.join('subset_mask.fits').strpath

    original_mask = [[0, 1], [1, 0]]
    hdu = fits.PrimaryHDU(data=original_mask)
    hdu.writeto(mask_filename)

    masks = fits_subset_mask_importer(mask_filename)

    assert len(masks) == 1
    label, mask = list(masks.items())[0]

    assert label == 'subset_mask'
    assert_equal(mask, original_mask)


def test_reader_extensions(tmpdir):

    mask_filename = tmpdir.join('subset_mask.fits').strpath

    original_mask1 = [[0, 1], [1, 0]]
    hdu1 = fits.ImageHDU(data=original_mask1, name='Subset A')
    original_mask2 = [[1, 0], [1, 0]]
    hdu2 = fits.ImageHDU(data=original_mask2)

    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
    hdulist.writeto(mask_filename)

    masks = fits_subset_mask_importer(mask_filename)

    assert len(masks) == 2
    mask_items = list(masks.items())
    label1, mask1 = mask_items[0]
    label2, mask2 = mask_items[1]

    assert label1 == 'SUBSET A'
    assert_equal(mask1, original_mask1)

    assert label2 == 'subset_mask[2]'
    assert_equal(mask2, original_mask2)


def test_reader_invalid_hdus(tmpdir):

    mask_filename = tmpdir.join('subset_mask.fits').strpath

    hdu = fits.PrimaryHDU(data=[1.2, 3.2, 4.5])
    hdu.writeto(mask_filename)

    with pytest.raises(ValueError) as exc:
        fits_subset_mask_importer(mask_filename)
    assert exc.value.args[0] == 'No HDUs with integer values (which would normally indicate a mask) were found in file'


def test_reader_invalid_format(tmpdir):

    mask_filename = tmpdir.join('subset_mask.fits').strpath

    with open(mask_filename, 'w') as f:
        f.write('qwdiwqoidqwjdijwq')

    with pytest.raises(IOError) as exc:
        fits_subset_mask_importer(mask_filename)
    assert exc.value.args[0].endswith('is not a valid FITS file')


def test_writer(tmpdir):

    mask_filename = tmpdir.join('subset_mask.fits').strpath

    masks = OrderedDict()
    masks['subset 1'] = [[0, 1], [1, 0]]
    masks['subset 2'] = [[0, 1, 3], [1, 0, 2]]

    fits_subset_mask_exporter(mask_filename, masks)

    with fits.open(mask_filename) as hdulist:

        assert len(hdulist) == 3

        assert hdulist[0].data is None

        assert hdulist[1].name == 'SUBSET 1'
        assert_equal(hdulist[1].data, masks['subset 1'])

        assert hdulist[2].name == 'SUBSET 2'
        assert_equal(hdulist[2].data, masks['subset 2'])
