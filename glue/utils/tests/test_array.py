import pytest
import numpy as np

from ...external.six import string_types

from ..array import view_shape, coerce_numeric, stack_view


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

    assert x is coerce_numeric(x)


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
