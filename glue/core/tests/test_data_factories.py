from distutils.version import LooseVersion

import pytest
from mock import MagicMock
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from astropy import __version__ as _astro_ver_
try:
    import astrodendro
    missing_astrodendro = False
except ImportError:
    missing_astrodendro = True

from .. import data_factories as df
from ..data import CategoricalComponent
from .util import make_file

needs_astropy_03 = pytest.mark.skipif(LooseVersion(_astro_ver_) < LooseVersion('0.3'),
                                      reason='Astropy >=0.3 feature')
needs_astrodendro = pytest.mark.skipif(missing_astrodendro, reason='Astrodendro feature')


def test_load_data():
    factory = MagicMock()
    result = MagicMock()
    result.label = ''
    factory.return_value = result
    d = df.load_data('test.fits', factory)
    factory.assert_called_once_with('test.fits')
    assert d.label == 'test'


def test_extension():
    assert df._extension('test.fits') == 'fits'
    assert df._extension('test.fits.gz') == 'fits.gz'
    assert df._extension('test.fits.gzip') == 'fits.gzip'
    assert df._extension('test.fits.bz') == 'fits.bz'
    assert df._extension('test.fits.bz2') == 'fits.bz2'
    assert df._extension('test.other.names.fits') == 'fits'


def test_data_label():
    assert df.data_label('test.fits') == 'test'
    assert df.data_label('/Leading/Path/test.fits') == 'test'
    assert df.data_label('') == ''
    assert df.data_label('/Leading/Path/no_extension') == 'no_extension'
    assert df.data_label('no_extension') == 'no_extension'


def test_png_loader():
    data = '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x00\x00\x00\x00W\xddR\xf8\x00\x00\x00\x0eIDATx\x9ccdddab\x04\x00\x00&\x00\x0b\x8e`\xe7A\x00\x00\x00\x00IEND\xaeB`\x82'
    with make_file(data, '.png') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.img_data
    assert_array_equal(d['PRIMARY'], [[3, 4], [1, 2]])


def test_fits_image_loader():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.gridded_data
    assert_array_equal(d['PRIMARY'], [1, 2, 3])


def test_fits_uses_mmapping():
    with make_file(TEST_FITS_DATA, '.fits', decompress=True) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.gridded_data
    assert not d['PRIMARY'].flags['OWNDATA']


@pytest.mark.parametrize('suffix', ['.h5', '.hdf5', '.hd5', '.h5custom'])
def test_hdf5_loader(suffix):
    data = 'x\xda\xeb\xf4pq\xe3\xe5\x92\xe2b\x00\x01\x0e\x0e\x06\x16\x06\x01\x06d\xf0\x1f\n*8P\xf90\xf9\x04(\xcd\x08\xa5;\xa0\xf4\n&\x988#XN\x02*.\x085\x1f]]H\x90\xab+H\xf5\x7f4\x00\xb3\xc7\x80\x05Bs0\x8c\x82\x91\x08<\\\x1d\x03@t\x04\x94\x0fK\xa5\'\x98P\xd5U\xa0\xa5G\x0f\n\xeded`\x83\x98\xc5\x08\xe3CR2##D\x80\x19\xaa\x0eA\x0b\x80\x95\np\xc0\xd2\xaa\x03\x98d\x05\xf2@\xe2LLL\x8c\x90t,\x01\xe633&@\x93\xb4\x04\x8a\xbdBP\xdd 5\xc9\xd5]A\x0c\x0c\r\x83"\x1e\x82\xfd\xfc]@9\x1a\x96\x0f\x15\x98G\xd3\xe6(\x18\x05\xa3\x00W\xf9\t\x01Lh\xe5$\x00\xc2A.\xaf'
    with make_file(data, suffix, decompress=True) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.gridded_data
    assert_array_equal(d['/x'], [1, 2, 3])


def test_fits_catalog_factory():
    data = '\x1f\x8b\x08\x08\x19\r\x9cQ\x02\x03test.fits\x00\xed\xd7AO\x830\x18\xc6\xf1\xe9\'yo\x1c\'\x1c\x8c\x97\x1d\x86c\xa6\x911"5\xc1c\x91n\x92\x8cBJ\x97\xb8o\xef\x06\xd3\x98H\xdd\x16\x97]|~\x17\x12H\xfeyI{h\x136\x8b\xc3\x80hD=8\r\xe9\xb5R\x8bJ\x97\r\x99\x8a\xa6\x8c\'\xd4\x18\xa1r\xa1s\xea\xe53\x1e\xb3\xd4\xd2\xbb\xdb\xf6\x84\xd6bC\xb90\x82\xcc\xa6\x96t@4NYB\x96\xde\xcd\xb6\xa7\xd6e&5U\x8b\xcfrQJ\xd5\x14\x95jz{A\xca\x83hb\xfd\xdf\x93\xb51\x00\x00\x00\x00\xf87v\xc7\xc9\x84\xcd\xa3\x119>\x8b\xf8\xd8\x0f\x03\xe7\xdb\xe7!e\x85\x12zCFd+I\xf2\xddt\x87Sk\xef\xa2\xe7g\xef\xf4\xf3s\xdbs\xfb{\xee\xed\xb6\xb7\x92ji\xdev\xbd\xaf\x12\xb9\x07\xe6\xf3,\xf3\xb9\x96\x9eg\xef\xc5\xf7\xf3\xe7\x88\x1fu_X\xeaj]S-\xb4(\xa5\x91\xba\xff\x7f\x1f~\xeb\xb9?{\xcd\x81\xf5\xe0S\x16\x84\x93\xe4\x98\xf5\xe8\xb6\xcc\xa2\x90\xab\xdc^\xe5\xfc%\x0e\xda\xf5p\xc4\xfe\x95\xf3\x97\xfd\xcc\xa7\xf3\xa7Y\xd7{<Ko7_\xbb\xbeNv\xb6\xf9\xbc\xf3\xcd\x87\xfb\x1b\x00\x00\xc0\xe5\r:W\xfb\xe7\xf5\x00\x00\x00\x00\x00\x00\xac>\x00\x04\x01*\xc7\xc0!\x00\x00'
    with make_file(data, '.fits') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.tabular_data

    assert_array_equal(d['a'], [1])
    assert_array_equal(d['b'], [2])


@pytest.mark.parametrize(('delim', 'suffix'),
                         ((',', '.csv'),
                          ('\t', '.tsv'),
                          ('|', '.txt'),
                          (' ', '.dat'),
                          ('\t', '.tbl')))
def test_ascii_catalog_factory(delim, suffix):
    data = "#a%sb\n1%s2" % (delim, delim)
    with make_file(data, suffix) as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.tabular_data

    assert_array_equal(d['a'], [1])
    assert_array_equal(d['b'], [2])


@pytest.mark.parametrize(('delim', 'suffix'),
                         ((',', '.csv'),
                          ('\t', '.tsv'),
                          ('|', '.txt'),
                          (' ', '.dat'),
                          ('\t', '.tbl')))
def test_pandas_parse_delimiters(delim, suffix):
    data = "a%sb\n1%s2" % (delim, delim)
    with make_file(data, suffix) as fname:
        d = df.load_data(fname, factory=df.pandas_read_table)

    assert_array_equal(d['a'], [1])
    assert_array_equal(d['b'], [2])


def test_fits_gz_factory():
    data = '\x1f\x8b\x08\x08\xdd\x1a}R\x00\x03test.fits\x00\xed\xd1\xb1\n\xc20\x10\xc6q\x1f\xe5\xde@ZA]\x1cZ\x8d\x10\xd0ZL\x87\xe2\x16m\x0b\x1d\x9aHR\x87n>\xba\xa5".\tRq\x11\xbe_\xe6\xfb\x93\xe3\x04\xdf\xa7;F\xb4"\x87\x8c\xa6t\xd1\xaa\xd2\xa6\xb1\xd4j\xda\xf2L\x90m\xa5*\xa4)\\\x03D1\xcfR\x9e\xbb{\xc1\xbc\xefIcdG\x85l%\xb5\xdd\xb5tW\xde\x92(\xe7\x82<\xff\x0b\xfb\x9e\xba5\xe7\xd2\x90\xae^\xe5\xba)\x95\xad\xb5\xb2\xfe^\xe0\xed\x8d6\xf4\xc2\xdf\xf5X\x9e\xb1d\xe3\xbd\xc7h\xb1XG\xde\xfb\x06_\xf4N\xecx Go\x16.\xe6\xcb\xf1\xbdaY\x00\x00\x00\x80?r\x9f<\x1f\x00\x00\x00\x00\x00|\xf6\x00\x03v\xd8\xf6\x80\x16\x00\x00'

    with make_file(data, '.fits.gz') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.gridded_data

    assert_array_equal(d['PRIMARY'], [[0, 0], [0, 0]])


def test_csv_gz_factory():
    data = '\x1f\x8b\x08\x08z\x1e}R\x00\x03test.csv\x00\xab\xe02\xe42\xe22\xe6\x02\x00y\xffzx\x08\x00\x00\x00'
    with make_file(data, '.csv.gz') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.tabular_data

    assert_array_equal(d['x'], [1, 2, 3])


@needs_astropy_03
def test_sextractor_factory():
    data = """#   1 NUMBER                 Running object number
#   2 X_IMAGE                Object position along x                                    [pixel]
#   3 Y_IMAGE                Object position along y                                    [pixel]
   1 2988.249    2.297
   2 2373.747    3.776
   3 3747.026    4.388"""
    with make_file(data, '.cat') as fname:
        d = df.load_data(fname, factory=df.sextractor_factory)
    assert_allclose(d['NUMBER'], [1, 2, 3])
    assert_allclose(d['X_IMAGE'], [2988.249, 2373.747, 3747.026])
    assert_allclose(d['Y_IMAGE'], [2.297, 3.776, 4.388])


def test_excel_factory():
    data = 'x\xda\xedX=h\x14A\x14\xfef\xf7\xfeI.\xbb\xe7EL\x84\xb0\x04\x8c\x1a\xd3\x04\x1b\x9bdOAS\x19\xa2\x16\x8a\x08z1\x0bJ\xc2E\x8e\x14\xc6\xc6h\xbcR\x10\xac\x14\x9b@\x1a\x9b\xa8\x8d?\x18A;\x0b!\xa2\x85 \x08w\n6V\x82B\x8a\xe4\xd67og\xcd%^q\x07\x1aT\xe6[\xe6\xcd\xdb7\xdf\xdby\xc7\xbcy{\xb3\xaf\x97\xec\xf2\xdc\x83\xce\n6`\x10&\xaa~\x12\xb1\x1a\x9b\xa0\x96\x0co,\xd0\xb8\xefK5\xec\x13\xd4|\x8d\x7f\n\xc9\x04-d,\x8a\xa7\xad\xaf\xe2r\r\xe5zW`\xe0~\xe4\x05I\xe0#\xb5S\xb8\x80\xe1\xc9\x82\xe7l"\x0ep\x0cy!c\x18 )p\x87,itpT\x19\x96gYnay\x8f\x99\x8b,sl\xb9\xcer\x80\xb8eq\x12K\xeep\xef>\x95\xc5\'\x8cn\x1eKC>\xf7\x11\xfb\xbcgK?\xda\xf1Rf\xf1\xe5\x1b"\xe0F\xb1\xbfx>?\xf1\x17\x0c\x08\x1aX4\x1a\xf3\xe8\x8a\xb4`\x1e\xb4\xa0C^\xc1+\xe6\'\xca\xc8\xd2\xca\xce\xe3\xbb\xef\x00\xdf\xc2-\xfc\xdc\xd1\xf6\xcd\xb5\x0b\x90}y\xbd=\xcevZ\xd8e\xdb\xce99\'\xb4\'\xea\xf0o\x1a\x11`\x06\xfe\x19\xde\x11%\xca\xe0\xcff\xb0k\x8f\x9d\xf3\xbc\xa9\xfe\x15*\xcb2E\x8c\xa0Z_\xa46\x9d\x92U\x9bw\xb9\xb5n\x97\xb7r\xf6\xb7\x90\x1cC\x1b\xeb6\xfbY4\xf3\xca\xdd\xafo\x0e\x8f\x8e\xb8\xa7\xd92\xc3\x95=\xa8\xff;d\x04\xf0qEz\x90s\x9aG",e(\xbd\xec\xb1\x87\xe5U~\xeav\xd6;Yf)A\xa9\xef\x19iW\xca\xa1Y\xe6\\\xe3\xd1\x1e\x9ag/\xe3\xad\xbb\xb3F\xdfEz\xe9\xcb\x91\xc7]\xa5O\xeen\xd2\x17\x86*\x97\xb2\x0b\xef\xdc9t\xd3\xfbh\x8c\xfc\xe55\x8b>\xd1\'n\xdf\x92x\xe2\x86\xbdP\xb5\xe2\x03\xcb\x8e_\xeaF\xc2\xb0T\xec\xbez\xc9\xb5a\x15)\xfe\xb56\x1b\x82;\x83\xee\x84\xe2\x8b\r\xfcg\xd8\xc66[\xd5\xa6\x18\xcb\xc0K\xf2\x8d:|\x83\xf9k\xcc\x0c\xd2?\xf9f\x1d\xbe\xc9\xfc5f\x86.\xc9\x1f42x\xc8U!W\xf3\xa6NACCCCCCCCC\xa3\x1e\x84:-\x98\xea_|T\x9d\x0e\xe2\xea\xbb\xce*\xb5\xaa\xfeL\xf2\xdf\xe2(&\xe9\x9a\xa2s\xe6A\x14\xa8/b\xba\xa9\xfc\xd9\x8a\xa8\x08\x9f%\x1a\xf4\t\xbf\x17J\x1c\xa7\xd9\x8b\x18\xc7(\xc71\xdet\xfe\xd2\xe9N\xd4\xfe\x9e\x86\x1d\xad\xdf\xb7\x85\x9a\x9d\xbf\xdaL\x9c\x7fx\xfe\x1f\xf5\x81\xcaV'
    with make_file(data, '.xlsx', decompress=True) as fname:
        d = df.load_data(fname)

    assert_array_equal(d['x'], [1, 2, 3])
    assert_array_equal(d['y'], [2, 3, 4])


def test_csv_pandas_factory():
    data = """a,b,c,d
1,2.1,some,True
2,2.4,categorical,False
3,1.4,data,True
4,4.0,here,True
5,6.3,,False
6,8.7,,False
8,9.2,,True"""

    with make_file(data, '.csv') as fname:
        d = df.load_data(fname, factory=df.pandas_read_table)
    assert d['a'].dtype == np.int
    assert d['b'].dtype == np.float
    assert d['c'].dtype == np.float
    cat_comp = d.find_component_id('c')
    assert isinstance(d.get_component(cat_comp), CategoricalComponent)
    correct_cats = np.unique(np.asarray(['some', 'categorical',
                                         'data', 'here',
                                         np.nan, np.nan, np.nan],
                                        dtype=np.object))
    np.testing.assert_equal(d.get_component(cat_comp)._categories,
                            correct_cats)
    cat_comp = d.find_component_id('d')
    assert isinstance(d.get_component(cat_comp), CategoricalComponent)


def test_dtype_int():
    data = '# a, b\n1, 1 \n2, 2 \n3, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == np.int


def test_dtype_float():
    data = '# a, b\n1., 1 \n2, 2 \n3, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == np.float


def test_dtype_float_on_categorical():
    data = '# a, b\nf, 1 \nr, 2 \nk, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == np.float


def test_dtype_badtext():
    data = '# a, b\nlabel1, 1 \n2, 2 \n3, 3\n4, 4\n5, 5\n6, 6'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == np.float
    assert_array_equal(d['a'], [np.nan, 2, 3, 4, 5, 6])


def test_dtype_missing_data_col2():
    data = '# a, b\n1 , 1 \n2,  \n3, 3.0'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['b'].dtype == np.float
    assert_array_equal(d['b'], [1, np.nan, 3])


def test_dtype_missing_data_col1():
    data = '# a, b\n1.0, 1 \n , 2 \n3, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == np.float
    assert_array_equal(d['a'], [1, np.nan, 3])


def test_column_spaces():
    data = '#a, b\nhere I go, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == np.float
    assert_array_equal(d['a'], [np.nan, 2, 3, 5, 7])


def test_casalike():
    data = 'x\xda\xed\x98Qo\xa3F\x10\xc7\xfbQ\xe6\xcd\x89\xce`v\xd9] R\x1f0\xde$$\xd8\xf8\x80Xq_*.!\xa9%\x1b\xa7@zM?}w\xc1v.\x97\xf8\xae\xc6\xfb\xd0\x07\xfe\xf2\x83e\xc9?\r\xf3\x1fvf6\xf6\xc7\xd3\x80\x03\xfc\n\x1f(\x81\x01\xdc\xad\xf3\x87u\xb1*\xa1Z\xc3\xb9\x9f\xc4PVi~\x9f\x16\xf7\xf0\xa1\x86~2\xf5o?\xe4i\x8c\x08^Z\x14\xe9\x0b\xdc\xa7U\n\xd5\xcbS\x06?\xd1\xc4\xbd\xf5c\xd8\x13\x9f\xe4\xe5\xcf\xab/Y\x01\xeb\x87-y\xb1\xca\xf2r\xb1\xce\xcb\xfd<\xb4\x87\x87\xe1`\xd5<\xac\x98g*\xe6\x91\xbd\xf9;X\xc3\xb1{\x05\x1b?\x98\xce,\xc2lF\x0cj:\x84kF+\x9e?y\xe5\x99\x14[\x84\xda\x98\x98\x8e\xd1\x927ua\xc3\xd3\x88Nm\xdbf\xb6i"\xc7p\xf8\'\x03\xb5\xe0%\xf3)\xafy=?\xafdeU/=h\xafpx\xc5\xbd\xa4\xe1y\x00&\xc1p\x0cN\xb9\x867\x13?i\x9e\xf7j>\x18rw\xfc&\xbe\xc1\xb0X<\xfeQ\xe5YY\xc2\xc9\xd3\xe2\xefly\n\xcf\xf9\xa2\xda\xcb\xe3\x9fo\xfcIx[\xfb\x8bu\xe3\x1b\t?\xcc\xc3\xe3\x8b\xdc\x11\x8f\xe7\xb1\x8c\xef\xfc\x9a\xd6?\x1d\x95\xbf \x9cLCq\x00\xca\xf8\x90n\xbf\x8d\xaf\xc5\xfb\x16\xb8\xc9\x8e\xc7\x04\xcfa\xc8\xa0\x94:\x94\xb5\xab\xbf\xa9g\x18H|\x9a\xf8\xbe\xcb\x9f\xd1\x8e\x877<C\x11\xcfT\xcc#\x8ay"\x7fX%\x0foxH]\xfe\xb0\xe2\xfca\xc5\xf93\x15\xe7\xcfT\\\x7f\xa6J?\x88\xe2\xf8D\xfe\x88\xe2\xfc\x11\xc5\xf9#\x8a\xeb\x8f\xa8\xf2\xc3\x93\xfd\xb7\x9e\xd7z\x91\xabiZ\xecO\x8e:\xef\xbdh\xe6\x06\x9b\xf9\x8f\x8a\xf9\xc5\xc0\x88 f!j\xb4;\x9f\xbd\x11\x0f\x92\x86\xa71\xdd!\xaf\x12\xf3\x0bm\x13\x9f\x18\x9fw\xf1!\xf3\xd8~\xe4\xc9~\xde\xe4\xef>{<\xbe_\xd6~\xd4\xf3no\xc4=E~`P\xd6/k?v<E~`\xc5~`\xc5~\xd4\xfbBo\x16N\x93\xe3y\xd2\x8ff\xff\xd0\xf0n\x1e\xc2\x04!\xf1\xbc\xb4\xa5\x1f\x9b}\xa69\x0f\x10c\x16\xc3\xd4\x12<\xd2\xd2\x8f7\xbc#\xcf\x17\xe9G\x93\xbf\xd5\xa0T\xe4G\xbdo\xf5\xe2$\xbc\xe6\xb1\n?\x88\xc2\xe7\x95~\xa8\xe4I?\x88b?\x9a\xfcm\x7f:*\x7f\xd3\x19\xfe\x1dm\xf6K%\xfdM\xf0\xb0B^\xc4\xe3\xe4<\xfa\\\xf3L\x9dP\x87\x99\xf6\x96\x87\x0c\x18DYY\xc1y\x91\xfd\xf9\x9c\xe5w/pr\xf9\xcf\xe9\x0fy\xf1\x94{\x9b\xfdh\xe8Fs\x8fO\x927\xfb[\xfc\x94\xddUE\xba\x84"{\xc8\n\xc1\xcc\xe0\xa1HW{\xefa\xdc \x91\x15\xb8\x8d\x8fa\xdb\xc2\x98"j\xa3&>wYeE\x9eV\x12\xb3\r\xf2\x15\xfdW\xba|\xce\xde\xf1\xe4\x05\xd1\x9ez\xf9\t\xaf\xde8\xdf\xf0f<\x88\xf8\xf9\xde\xfb\x92\x01\x82 \x8e\xfa\xe2\xdb%\x0f\xfa`\x8a\xfd;\xee\xc3\'L\x19D\xe9\xfdb\xfd\xbe\xfe\xc2\xf1X\xe4\x0c\xee\xd22\xbd[\x17\x19\xe4\xeb\\\xdb]y=\x97\xe9cv\x06D@G}\xa0p\xc1\xc3>0\x88\xc3\x9b>Xp!\xf2\xf4\xee\n\x8d\x07<\xf6\xc2\xa9\xf0\x83\xcf\x02\xf7\xe8z\x16\xf1\xf3h\xc6#\xd9\x7f\x0b\x1d\xdc\x95\x8c\x0c\xdc\xdd\xb7\xeb\xeci\x99\x1dpC1r\x13\xae\t\xa8\xe0a\x03!M|\xb0\x95\x18\xe8\x8c\x18g\x86\xa5\xd3\xda\x9a\x03"N\xfc\xf1v?\xbfI<P\xf1\xbc\x91\x0b\xea\xe65\xc1\x13s\x8b\xbayC\xf0D\x11h\xb7\xb2_"\x9d\x89\x14R\xd1\xdfL\xf1\x9a\x08\x1ek\xcd\x9bK\x1e\xd5\r\x82\x1c\xdbvlf0v\x14\xef\xb7\xe6\xfd\xa5\x94\xd8\x96\x83Mfc\xd4\x8e\'\xeb\xa5\xb9_\x93\xf5\x82\xc5L%J&A\xd6\x191\xcf(\xd6\x19\xa5u\xbd\x0cF\xf2\x15\xae\xef\x8b\x1f\x16\xcb\x0c\xbe\xa6%|-\x16U\x95\xe5\xdf\xc5\x17\xf9\x17\xf2\x06P\xf0<7v\xe5\x19\xa3\x1bp"\xde\xb8/\xe2o\x05r\x18\xc3\xa7\x07T\xd0\xa5/\xdan4\x87\x1a\x16\'n\x94@\x10^$\xee0\xe0\xad\xea\x8fOF\xd0\xa9S\xa7N\x9d:u\xea\xd4\xa9\xd3\xffE\xbft\xea\xd4\xa9S\xa7N\x9d:u\xea\xd4\xe9?\xeb_\xdc?$\x07'
    with make_file(data, '.fits', decompress=True) as fname:
        assert df.is_casalike(fname)
        d = df.load_data(fname, factory=df.casalike_cube)

    assert d.shape == (1, 2, 2, 2)
    d['STOKES 0']
    d['STOKES 1']
    d['STOKES 2']
    d['STOKES 3']

# zlib-compressed
TEST_FITS_DATA = 'x\x9c\xed\xd0\xb1\n\xc20\x14\x85\xe1\xaa/r\xde@\x8a\xe2\xe6\xa0X!\xa0\xa5\xd0\x0c]\xa3m\xa1C\x13I\xe2\xd0\xb7\xb7b\xc5\xa1)\xe2\xe6p\xbe\xe5N\xf7\xe7rsq\xceN\t\xb0E\x80\xc4\x12W\xa3kc[\x07op\x142\x87\xf3J\x97\xca\x96\xa1\x05`/d&\x8apo\xb3\xee{\xcaZ\xd5\xa1T^\xc1w\xb7*\\\xf9Hw\x85\xc81q_\xdc\xf7\xf4\xbd\xbdT\x16\xa6~\x97\x9b\xb6\xd2\xae1\xdaM\xf7\xe2\x89\xde\xea\xdb5cI!\x93\xf40\xf9\xbf\xdf{\xcf\x18\x11\x11\x11\x11\xfd\xad\xe8e6\xcc\xf90\x17\x11\x11\x11\x11\x11\x11\x8d<\x00\x7fy\x7f\xbc'


def test_data_reload():
    data = '#a, b\n0, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
        coords_old = d.coords
        with open(fname, 'w') as f2:
            f2.write('#a, b\n0, 0\n0, 0\n0, 0\n0, 0\n0, 0')
        d._load_log.reload()

    assert_array_equal(d['a'], [0, 0, 0, 0, 0])
    assert_array_equal(d['b'], [0, 0, 0, 0, 0])
    assert d.coords is not coords_old


def test_data_reload_no_file():
    data = '#a, b\n0, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)

    # file no longer exists
    d._load_log.reload()

    assert_array_equal(d['a'], [0, 2, 3, 5, 7])


def test_data_reload_shape_change():

    data = '#a, b\n0, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
        coords_old = d.coords
        with open(fname, 'w') as f2:
            f2.write('#a, b\n0, 0\n0, 0\n0, 0\n0, 0')
        d._load_log.reload()

    assert_array_equal(d['a'], [0, 2, 3, 5, 7])
    assert d.coords is coords_old


def test_file_watch():
    cb = MagicMock()
    with make_file('test', 'csv') as fname:
        fw = df.FileWatcher(fname, cb)
        fw.check_for_changes()
        assert cb.call_count == 0

        # fudge stat_cache to simulate filechange
        # we could just change the file, but
        # the underlying OS check has low time resolution
        # and would require a sleep
        fw.stat_cache -= 1

        fw.check_for_changes()
        assert cb.call_count == 1


def test_file_watch_os_error():
    cb = MagicMock()
    with make_file('test', 'csv') as fname:
        fw = df.FileWatcher(fname, cb)

    fw.check_for_changes()
    assert cb.call_count == 0


@needs_astrodendro
def test_dendrogram_load():
    data = """x\xda\xed\xda]K\xc2`\x18\xc6\xf1^\xbe\xc8}fA\xe4[X\x14\x1eX\x99<\x90S\xd8\x02O\x9f\xf2Q<\xd8&\xcf&\xe4\xb7\xcft\x82\xc9\xe6\x1be\x91\xff\xdf\xc9\xc5\xd8v\xc1vt\xeff\xaej\xb6\x9f\xeb"UI\xe1I^\xde\xc2\xa0\x17Z?\x928\x94\'\xe5\xb9\x12\xc5:\xe8j\xdb\x95T\xf7\xcak\xabNF\xdf\xcd\xa4O[\xab\xc7\xd2\xd5\xb1\x96x<4\xb2\x86S\xeb(W2\xfa\n\x93\xbe`\xe4\xbf\x1a+ao\xde<\xf0M\x10\r\xc2 J\xed\xabw\xbc\xba\xf3\x98\xf9\xbc[\x9b\x96\x01\x00\x00\xe0`|\x8e\x93\xaej9U\xc9\xa9f\xad1\x99\xa4%\xb7p:/\xca\xd7}#\xe6=\x9eM\xa5\xeb\xfaV\xcd\xcf\x95\xabo\x9e\x9f\x8b\xdb\xcf\xcf\xd3\xbebF_e\xfb\xf7\xd7~h\xbd8\xdeF\xf3\xfdP[\xed\x9b\xd8\xd8hE_cU\xdf\xd7\xe7\xed\xdbp4\x8c\x98\xef\x01\x00\x00\xf6\xeah\xe68\xc9\x93$O3\x8e\xe7\xd7\x01\x00\x00\x00\x07i\x9f\xfb\xe7r\x89\xfd3\xfbg\x00\x00\x80\x7f\xb1\x7fN\xdbA\x03\x00\x00\x00\xf8\xc5\xfd\xf3_\xff\xff\xb9t\xcd\xfe\x19\x00\x00\x00\x1b\xed\x9f\xcf\x96\xb2\x98\xe4m\x92\xe5$/\x93,d\xe4E\x92\xa5\x1d\xef?_:\xde\xf5\xfe;\xbe\x8c\x00\x00\x00\xf0\x13>\x00\x8e\xbe x"""
    with make_file(data, 'fits', decompress=True) as fname:
        dg, im = df.load_data(fname, factory=df.load_dendro)
    assert_array_equal(im['intensity'], [1, 2, 3, 2, 3, 1])
    assert_array_equal(im['structure'], [0, 0, 1, 0, 2, 0])
    assert_array_equal(dg['parent'], [-1, 0, 0])
    assert_array_equal(dg['height'], [3, 3, 3])
    assert_array_equal(dg['peak'], [3, 3, 3])
