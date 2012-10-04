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


def test_default_factories():
    for ext in ['fits', 'hdf5', 'hd5']:
        assert df.get_default_factory(ext) == df.gridded_data

    for ext in ['vot', 'csv', 'txt', 'tsv']:
        assert df.get_default_factory(ext) == df.tabular_data

    for ext in ['png', 'jpeg', 'jpg', 'bmp', 'tiff']:
        assert df.get_default_factory(ext) == df.pil_data

    assert df.get_default_factory('weird_extension') is None
