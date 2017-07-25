from __future__ import absolute_import, division, print_function

import os
from mock import patch

from numpy.testing import assert_equal
from astropy.io import fits

from glue.core import DataCollection, Data
from glue.io.qt.subset_mask import QtSubsetMaskImporter, QtSubsetMaskExporter


def test_importer(tmpdir):

    filename = tmpdir.join('test.fits').strpath

    hdu = fits.PrimaryHDU(data=[0, 1, 1])
    hdu.writeto(filename)

    data = Data(x=[1, 2, 3])
    data_collection = DataCollection([data])

    with patch('qtpy.compat.getopenfilename') as o:
        o.return_value = filename, 'FITS (*.fits *.fit *.fits.gz *.fit.gz)'
        importer = QtSubsetMaskImporter()
        importer.run(data, data_collection)

    assert_equal(data.subsets[0].to_mask(), [0, 1, 1])


def test_importer_cancel(tmpdir):

    filename = tmpdir.join('test.fits').strpath

    hdu = fits.PrimaryHDU(data=[0, 1, 1])
    hdu.writeto(filename)

    data = Data(x=[1, 2, 3])
    data_collection = DataCollection([data])

    with patch('qtpy.compat.getopenfilename') as o:
        o.return_value = '', ''  # simulates cancelling
        importer = QtSubsetMaskImporter()
        importer.run(data, data_collection)

    assert len(data_collection.subset_groups) == 0
    assert len(data.subsets) == 0


def test_exporter(tmpdir):

    filename = tmpdir.join('test.fits').strpath

    data = Data(x=[1, 2, 3])
    data_collection = DataCollection([data])
    data_collection.new_subset_group(subset_state=data.id['x'] >= 2, label='subset a')

    with patch('qtpy.compat.getsavefilename') as o:
        o.return_value = filename, 'FITS (*.fits *.fit *.fits.gz *.fit.gz)'
        exporter = QtSubsetMaskExporter()
        exporter.run(data)

    with fits.open(filename) as hdulist:
        assert len(hdulist) == 2
        assert hdulist[0].data is None
        assert hdulist[1].name == 'SUBSET A'
        assert_equal(hdulist[1].data, [0, 1, 1])


def test_exporter_cancel(tmpdir):

    filename = tmpdir.join('test.fits').strpath

    data = Data(x=[1, 2, 3])
    data_collection = DataCollection([data])
    data_collection.new_subset_group(subset_state=data.id['x'] >= 2, label='subset a')

    with patch('qtpy.compat.getsavefilename') as o:
        o.return_value = '', ''  # simulates cancelling
        exporter = QtSubsetMaskExporter()
        exporter.run(data)

    assert not os.path.exists(filename)
