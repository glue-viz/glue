import os
import tempfile
from contextlib import contextmanager

import pytest
from mock import MagicMock
import numpy as np

import glue.core.data_factories as df


@contextmanager
def make_file(contents, suffix):
    """Context manager to write data to a temporary file,
    and delete on exit

    :param contents: Data to write. string
    :param suffix: File suffix. string
    """
    try:
        fobj, fname = tempfile.mkstemp(suffix=suffix)
        with open(fname, 'wb') as infile:
            infile.write(contents)
        yield fname
    finally:
        os.unlink(fname)


def test_load_data():
    factory = MagicMock()
    d = df.load_data('test.fits', factory)
    factory.assert_called_once_with('test.fits')
    assert d.label == 'test'


def test_data_label():
    assert df.data_label('test.fits') == 'test'
    assert df.data_label('/Leading/Path/test.fits') == 'test'
    assert df.data_label('') == ''
    assert df.data_label('/Leading/Path/no_extension') == 'no_extension'
    assert df.data_label('no_extension') == 'no_extension'


def test_default_factory_gridded():
    for ext in ['fits', 'hdf5', 'hd5']:
        assert df.get_default_factory(ext) == df.gridded_data


def test_default_factory_tabular():
    for ext in ['vot', 'csv', 'txt', 'tsv', 'xml']:
        assert df.get_default_factory(ext) == df.tabular_data


#Factory is optional, and depends on PIL
@pytest.mark.skipif("not hasattr(df, 'pil_data')")
def test_default_factory_pil():
    for ext in ['png', 'jpeg', 'jpg', 'bmp', 'tiff']:
        assert df.get_default_factory(ext) == df.pil_data


def test_default_factory_notfound():
    assert df.get_default_factory('weird_extension') is None


def test_png_loader():
    data = '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x00\x00\x00\x00W\xddR\xf8\x00\x00\x00\x0eIDATx\x9ccdddab\x04\x00\x00&\x00\x0b\x8e`\xe7A\x00\x00\x00\x00IEND\xaeB`\x82'
    with make_file(data, '.png') as fname:
        d = df.load_data(fname)
    np.testing.assert_array_equal(d['PRIMARY'], [[3, 4], [1, 2]])


def test_fits_image_loader():
    data = '\x1f\x8b\x08\x08\xac\xf6\x9bQ\x00\x03test.fits\x00\xed\xd0\xb1\n\xc20\x14\x85\xe1\xaa/r\xde@\x8a\xe2\xe6\xa0X!\xa0\xa5\xd0\x0c]\xa3m\xa1C\x13I\xe2\xd0\xb7\xb7b\xc5\xa1)\xe2\xe6p\xbe\xe5N\xf7\xe7rsq\xceN\t\xb0E\x80\xc4\x12W\xa3kc[\x07op\x142\x87\xf3J\x97\xca\x96\xa1\x05`/d&\x8apo\xb3\xee{\xcaZ\xd5\xa1T^\xc1w\xb7*\\\xf9Hw\x85\xc81q_\xdc\xf7\xf4\xbd\xbdT\x16\xa6~\x97\x9b\xb6\xd2\xae1\xdaM\xf7\xe2\x89\xde\xea\xdb5cI!\x93\xf40\xf9\xbf\xdf{\xcf\x18\x11\x11\x11\x11\xfd\xad\xe8e6\xcc\xf90\x17\x11\x11\x11\x11\x11\x11\x8d<\x00\x8d,\xdc\xe8\x80\x16\x00\x00'
    with make_file(data, '.fits') as fname:
        d = df.load_data(fname)
    np.testing.assert_array_equal(d['PRIMARY'], [1, 2, 3])


def test_fits_catalog_factory():
    data = '\x1f\x8b\x08\x08\x19\r\x9cQ\x02\x03test.fits\x00\xed\xd7AO\x830\x18\xc6\xf1\xe9\'yo\x1c\'\x1c\x8c\x97\x1d\x86c\xa6\x911"5\xc1c\x91n\x92\x8cBJ\x97\xb8o\xef\x06\xd3\x98H\xdd\x16\x97]|~\x17\x12H\xfeyI{h\x136\x8b\xc3\x80hD=8\r\xe9\xb5R\x8bJ\x97\r\x99\x8a\xa6\x8c\'\xd4\x18\xa1r\xa1s\xea\xe53\x1e\xb3\xd4\xd2\xbb\xdb\xf6\x84\xd6bC\xb90\x82\xcc\xa6\x96t@4NYB\x96\xde\xcd\xb6\xa7\xd6e&5U\x8b\xcfrQJ\xd5\x14\x95jz{A\xca\x83hb\xfd\xdf\x93\xb51\x00\x00\x00\x00\xf87v\xc7\xc9\x84\xcd\xa3\x119>\x8b\xf8\xd8\x0f\x03\xe7\xdb\xe7!e\x85\x12zCFd+I\xf2\xddt\x87Sk\xef\xa2\xe7g\xef\xf4\xf3s\xdbs\xfb{\xee\xed\xb6\xb7\x92ji\xdev\xbd\xaf\x12\xb9\x07\xe6\xf3,\xf3\xb9\x96\x9eg\xef\xc5\xf7\xf3\xe7\x88\x1fu_X\xeaj]S-\xb4(\xa5\x91\xba\xff\x7f\x1f~\xeb\xb9?{\xcd\x81\xf5\xe0S\x16\x84\x93\xe4\x98\xf5\xe8\xb6\xcc\xa2\x90\xab\xdc^\xe5\xfc%\x0e\xda\xf5p\xc4\xfe\x95\xf3\x97\xfd\xcc\xa7\xf3\xa7Y\xd7{<Ko7_\xbb\xbeNv\xb6\xf9\xbc\xf3\xcd\x87\xfb\x1b\x00\x00\xc0\xe5\r:W\xfb\xe7\xf5\x00\x00\x00\x00\x00\x00\xac>\x00\x04\x01*\xc7\xc0!\x00\x00'
    with make_file(data, '.fits') as fname:
        d = df.load_data(fname, df.tabular_data)

    np.testing.assert_array_equal(d['a'], [1])
    np.testing.assert_array_equal(d['b'], [2])


@pytest.mark.parametrize(('delim', 'suffix'),
                         ((',', '.csv'),
                          ('\t', '.tsv'),
                          ('|', '.txt'),
                          (' ', '.dat'),
                          ('\t', '.tbl')))
def test_ascii_catalog_factory(delim, suffix):
    data = "#a%sb\n1%s2" % (delim, delim)
    print data
    with make_file(data, suffix) as fname:
        d = df.load_data(fname)

    np.testing.assert_array_equal(d['a'], [1])
    np.testing.assert_array_equal(d['b'], [2])
