import pytest
from mock import MagicMock

import glue.core.data_factories as df


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
