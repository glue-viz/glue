import pytest

from numpy.testing import assert_equal
from pandas import DataFrame

from glue.core import Data, DataCollection


def test_translator_data_roundtrip():

    # Case where we start from a Dataframe, convert to a glue data object,
    # and convert back to a DataFrame.

    df = DataFrame()
    df['a'] = [3, 5, 6, 7]
    df['b'] = [1.5, 2.2, 1.3, 3.3]
    df['c'] = ['r', 'd', 'w', 'q']

    dc = DataCollection()
    dc['dataframe'] = df

    data = dc['dataframe']
    assert isinstance(data, Data)
    assert data.main_components == ['a', 'b', 'c']
    assert_equal(data['a'], [3, 5, 6, 7])
    assert_equal(data['b'], [1.5, 2.2, 1.3, 3.3])
    assert_equal(data['c'], ['r', 'd', 'w', 'q'])

    df2 = data.get_object()
    assert_equal(list(df2.columns), ['a', 'b', 'c'])
    assert_equal(df2['a'].values, [3, 5, 6, 7])
    assert_equal(df2['b'].values, [1.5, 2.2, 1.3, 3.3])
    assert_equal(df2['c'].values, ['r', 'd', 'w', 'q'])


def test_translator_from_data():

    # Case where the initial data object wasn't originally created from a
    # DataFrame.

    data = Data()
    data['a'] = [3, 5, 6, 7]
    data['b'] = [1.5, 2.2, 1.3, 3.3]
    data['c'] = ['r', 'd', 'w', 'q']

    with pytest.raises(ValueError) as exc:
        df = data.get_object()
    assert exc.value.args[0] == ('Specify the object class to use with cls= - supported '
                                 'classes are:\n\n* pandas.core.frame.DataFrame')

    df = data.get_object(cls=DataFrame)
    assert_equal(list(df.columns), ['a', 'b', 'c'])
    assert_equal(df['a'].values, [3, 5, 6, 7])
    assert_equal(df['b'].values, [1.5, 2.2, 1.3, 3.3])
    assert_equal(df['c'].values, ['r', 'd', 'w', 'q'])


def test_translator_from_subset():

    # Case where we convert a subset to a DataFrame

    data = Data()
    data['a'] = [3, 5, 6, 7]
    data['b'] = [1.5, 2.2, 1.3, 3.3]
    data['c'] = ['r', 'd', 'w', 'q']

    dc = DataCollection([data])
    dc.new_subset_group(label='test subset', subset_state=data.id['b'] > 2)

    df = data.get_subset_object(cls=DataFrame)
    assert_equal(list(df.columns), ['a', 'b', 'c'])
    assert_equal(df['a'].values, [5, 7])
    assert_equal(df['b'].values, [2.2, 3.3])
    assert_equal(df['c'].values, ['d', 'q'])


def test_translator_from_data_with_derived():

    # Case where we convert a subset to a DataFrame

    data = Data()
    data['a'] = [3, 5, 6, 7]
    data['b'] = data.id['a'] + 1

    dc = DataCollection([data])
    dc.new_subset_group(label='test subset', subset_state=data.id['b'] > 2)

    df = data.get_object(cls=DataFrame)
    assert_equal(list(df.columns), ['a', 'b'])
    assert_equal(df['a'].values, [3, 5, 6, 7])
    assert_equal(df['b'].values, [4, 6, 7, 8])


# TODO: Case where we convert from a BaseData object
