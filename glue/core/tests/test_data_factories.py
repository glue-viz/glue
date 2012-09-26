from glue.core.data_factories import data_label

def test_data_label():
    assert data_label('test.fits') == 'test'
    assert data_label('/Leading/Path/test.fits') == 'test'
    assert data_label('') == ''
    assert data_label('/Leading/Path/no_extension') == 'no_extension'
    assert data_label('no_extension') == 'no_extension'
