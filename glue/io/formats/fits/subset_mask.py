import os

from astropy.io import fits
from glue.config import subset_mask_importer, subset_mask_exporter
from glue.core.data_factories.fits import is_fits


@subset_mask_importer(label='FITS', extension=['fits', 'fit',
                                               'fits.gz', 'fit.gz'])
def fits_subset_mask_importer(filename):

    if not is_fits(filename):
        raise IOError("File {0} is not a valid FITS file".format(filename))

    masks = {}

    label = os.path.basename(filename).rpartition('.')[0]

    with fits.open(filename) as hdulist:

        masks = {}

        for ihdu, hdu in enumerate(hdulist):
            if hdu.data is not None and hdu.data.dtype.kind == 'i':
                if hdu.name is None:
                    name = '{0}[{1}]'.format(label, ihdu)
                else:
                    name = hdu.name
                masks[name] = hdu.data > 0

    if len(masks) == 0:
        raise ValueError('No HDUs with integer values were found in file')

    return masks


@subset_mask_exporter(label='FITS', extension=['fits', 'fit',
                                               'fits.gz', 'fit.gz'])
def fits_subset_mask_exporter(filename, masks):

    hdulist = fits.HDUList()
    for label, mask in masks.items():
        hdulist.append(fits.ImageHDU(mask.astype(int), name=label))

    hdulist.writeto(filename, overwrite=True)
