import pytest
import numpy as np

from glue.external.six import string_types, PY2

from ..array import view_shape, coerce_numeric, stack_view, unique, shape_to_string, check_sorted


@pytest.mark.parametrize(('before', 'ref_after', 'ref_indices'),
                         (([2.2, 5, 4, 4, 2, 8.3, 2.2], [2, 2.2, 4, 5, 8.3], [1, 3, 2, 2, 0, 4, 1]),
                          ([2.2, 5, np.nan, 2, 8.3, 2.2], [2, 2.2, 5, 8.3], [1, 2, -1, 0, 3, 1])))
def test_unique(before, ref_after, ref_indices):
    after, indices = unique(before)
    np.testing.assert_array_equal(after, ref_after)
    np.testing.assert_array_equal(indices, ref_indices)


def test_shape_to_string():
    assert shape_to_string((1, 4, 3)) == "(1, 4, 3)"


@pytest.mark.skipif("not PY2")
def test_shape_to_string_long():
    # Shape includes long ints on Windows
    assert shape_to_string((long(1), long(4), long(3))) == "(1, 4, 3)"


def test_view_shape():
    assert view_shape((10, 10), np.s_[:]) == (10, 10)
    assert view_shape((10, 10, 10), np.s_[:]) == (10, 10, 10)
    assert view_shape((10, 10), np.s_[:, 1]) == (10,)
    assert view_shape((10, 10), np.s_[2:3, 2:3]) == (1, 1)
    assert view_shape((10, 10), None) == (10, 10)
    assert view_shape((10, 10), ([1, 2, 3], [2, 3, 4])) == (3,)


def test_coerce_numeric():

    x = np.array(['1', '2', '3.14', '4'], dtype=str)
    np.testing.assert_array_equal(coerce_numeric(x),
                                  [1, 2, 3.14, 4])

    x = np.array([1, 2, 3])
    assert coerce_numeric(x) is x

    x = np.array([0, 1, 1, 0], dtype=bool)
    np.testing.assert_array_equal(coerce_numeric(x), np.array([0, 1, 1, 0], dtype=np.int))


@pytest.mark.parametrize(('shape', 'views'),
                         [
                             [(5, 5), (np.s_[0:3],)],
                             [(5, 4), (np.s_[0:3],)],
                             [(5, 4), ((3, 2),)],
                             [(5, 4), (np.s_[0:4], np.s_[:, 0:2])],
                             [(5, 4), (np.s_[0:3, 0:2], 'transpose', (0, 0))],
                             [(10, 20), (np.random.random((10, 20)) > 0.1, 3)],
                             [(5, 7), ('transpose', (3, 2))],
])
def test_stack_view(shape, views):
    x = np.random.random(shape)
    exp = x
    for v in views:
        if isinstance(v, string_types) and v == 'transpose':
            exp = exp.T
        else:
            exp = exp[v]

    actual = x[stack_view(shape, *views)]

    np.testing.assert_array_equal(exp, actual)


@pytest.mark.parametrize(('array', 'is_sorted'),
                         (([1, 3, 4, 3], False), ([1, 2, np.nan, 3], True), ([1, 3, 4, 4.1], True)))
def test_check_sorted(array, is_sorted):
    assert check_sorted(array) is is_sorted
