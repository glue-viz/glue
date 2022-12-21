import sys
import pytest
import numpy as np
from unittest.mock import MagicMock
from numpy.testing import assert_allclose, assert_array_equal

from glue.core.component import CategoricalComponent
from glue.core.data import BaseCartesianData, Data
from glue.core import data_factories as df
from glue.config import data_factory
from glue.tests.helpers import requires_astropy, make_file, requires_qt


def test_load_data_auto_assigns_label():
    factory = MagicMock()
    result = Data(x=[1, 2, 3], label='')
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


@pytest.mark.parametrize(('delim', 'suffix'),
                         ((',', '.csv'),
                          ('\t', '.tsv'),
                          ('|', '.txt'),
                          (' ', '.dat'),
                          ('\t', '.tbl')))
def test_ascii_catalog_factory(delim, suffix):
    data = ("#a%sb\n1%s2" % (delim, delim)).encode('ascii')
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
    data = ("a%sb\n1%s2" % (delim, delim)).encode('ascii')
    with make_file(data, suffix) as fname:
        d = df.load_data(fname, factory=df.pandas_read_table)

    assert_array_equal(d['a'], [1])
    assert_array_equal(d['b'], [2])


@requires_astropy
def test_csv_gz_factory():
    data = b'\x1f\x8b\x08\x08z\x1e}R\x00\x03test.csv\x00\xab\xe02\xe42\xe22\xe6\x02\x00y\xffzx\x08\x00\x00\x00'
    with make_file(data, '.csv.gz') as fname:
        d = df.load_data(fname)
        assert df.find_factory(fname) is df.tabular_data

    assert_array_equal(d['x'], [1, 2, 3])


@requires_astropy
def test_sextractor_factory():
    data = b"""#   1 NUMBER                 Running object number
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


def test_csv_pandas_factory():
    data = b"""a,b,c,d
1,2.1,some,True
2,2.4,categorical,False
3,1.4,data,True
4,4.0,here,True
5,6.3,,False
6,8.7,,False
8,9.2,,True"""

    with make_file(data, '.csv') as fname:
        d = df.load_data(fname, factory=df.pandas_read_table)
    assert d['a'].dtype == np.int64
    assert d['b'].dtype == float
    assert d['c'].dtype.kind == 'U'
    cat_comp = d.find_component_id('c')
    assert isinstance(d.get_component(cat_comp), CategoricalComponent)
    correct_cats = np.unique(np.asarray(['some', 'categorical',
                                         'data', 'here',
                                         '', '', '']))
    np.testing.assert_equal(d[cat_comp].categories, correct_cats)
    cat_comp = d.find_component_id('d')
    assert isinstance(d.get_component(cat_comp), CategoricalComponent)


def test_dtype_int():
    data = b'# a, b\n1, 1 \n2, 2 \n3, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == int


def test_dtype_float():
    data = b'# a, b\n1., 1 \n2, 2 \n3, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == float


def test_dtype_str_on_categorical():
    data = b'# a, b\nf, 1 \nr, 2 \nk, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype.kind in 'SU'


def test_dtype_badtext():
    data = b'# a, b\nlabel1, 1 \n2, 2 \n3, 3\n4, 4\n5, 5\n6, 6'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == float
    assert_array_equal(d['a'], [np.nan, 2, 3, 4, 5, 6])


def test_dtype_missing_data_col2():
    data = b'# a, b\n1 , 1 \n2,  \n3, 3.0'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['b'].dtype == float
    assert_array_equal(d['b'], [1, np.nan, 3])


def test_dtype_missing_data_col1():
    data = b'# a, b\n1.0, 1 \n , 2 \n3, 3'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == float
    assert_array_equal(d['a'], [1, np.nan, 3])


def test_column_spaces():
    data = b'#a, b\nhere I go, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
    assert d['a'].dtype == float
    assert_array_equal(d['a'], [np.nan, 2, 3, 5, 7])


def test_data_reload():
    data = b'#a, b\n0, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
        coords_old = d.coords
        with open(fname, 'w') as f2:
            f2.write('#a, b\n0, 0\n0, 0\n0, 0\n0, 0\n0, 0')
        d._load_log.reload()

    assert_array_equal(d['a'], [0, 0, 0, 0, 0])
    assert_array_equal(d['b'], [0, 0, 0, 0, 0])


@pytest.mark.skipif(sys.platform.startswith('win'), reason='file deletion doesn\'t work on Windows')
def test_data_reload_no_file():
    data = b'#a, b\n0, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)

    # file no longer exists
    with pytest.warns(UserWarning, match='Could not reload'):
        d._load_log.reload()

    assert_array_equal(d['a'], [0, 2, 3, 5, 7])


def test_data_reload_shape_change():

    data = b'#a, b\n0, 1\n2, 3\n3, 4\n5, 6\n7, 8'
    with make_file(data, '.csv') as fname:
        d = df.load_data(fname)
        coords_old = d.coords
        with open(fname, 'w') as f2:
            f2.write('#a, b\n0, 0\n0, 0\n0, 0\n0, 0')

        with pytest.warns(UserWarning, match='Cannot refresh data -- data shape changed'):
            d._load_log.reload()

    assert_array_equal(d['a'], [0, 2, 3, 5, 7])
    assert d.coords is coords_old


# TODO: this doesn't belong in the core since it relies on Qt
@requires_qt
def test_file_watch():
    cb = MagicMock()
    with make_file(b'test', 'csv') as fname:
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


@requires_qt
@pytest.mark.skipif(sys.platform.startswith('win'), reason='file deletion doesn\'t work on Windows')
def test_file_watch_os_error():
    cb = MagicMock()
    with make_file(b'test', 'csv') as fname:
        fw = df.FileWatcher(fname, cb)

    with pytest.warns(UserWarning, match='Cannot access'):
        fw.check_for_changes()

    assert cb.call_count == 0


def test_ambiguous_format(tmpdir):

    @data_factory('b', identifier=df.has_extension('spam'), priority=34)
    def reader1(filename):
        return Data()

    @data_factory('a', identifier=df.has_extension('spam'), priority=34)
    def reader2(filename):
        return Data()

    @data_factory('c', identifier=df.has_extension('spam'), priority=22)
    def reader3(filename):
        return Data()

    filename = tmpdir.join('test.spam').strpath
    with open(filename, 'w') as f:
        f.write('Camelot!')

    # Should raise a warning and pick the highest priority one in alphabetical
    # order

    with pytest.warns(UserWarning, match="Multiple data factories matched the input: 'a', 'b'. Choosing 'a'."):
        factory = df.find_factory(filename)

    assert factory is reader2


def test_basedata(tmpdir):

    # Regression test for a bug that caused load_data to fail if a data
    # factory returned a BaseData (but not Data) subclass, due to the
    # LoadLog expecting a Data object

    class BigData(BaseCartesianData):

        def get_kind(self):
            pass

        def compute_histogram(self):
            pass

        def compute_statistic(self):
            pass

        def get_mask(self):
            pass

        @property
        def shape(self):
            pass

        @property
        def main_components(self):
            return []

    @data_factory('bigdata', identifier=df.has_extension('bigdata'), priority=34)
    def reader(filename):
        return BigData()

    filename = tmpdir.join('test.bigdata').strpath
    with open(filename, 'w') as f:
        f.write('Camelot!')

    d = df.load_data(filename)

    assert isinstance(d, BigData)
