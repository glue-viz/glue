from __future__ import absolute_import, division, print_function

from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from glue.tests.helpers import HYPOTHESIS_INSTALLED
from glue.external.six import string_types, PY2  # noqa

from ..array import (view_shape, coerce_numeric, stack_view, unique, broadcast_to,
                     shape_to_string, check_sorted, pretty_number, unbroadcast,
                     iterate_chunks, combine_slices, nanmean, nanmedian, nansum,
                     nanmin, nanmax, format_minimal)


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
    assert shape_to_string((long(1), long(4), long(3))) == "(1, 4, 3)"  # noqa


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


class TestPrettyNumber(object):

    def test_single(self):
        assert pretty_number([1]) == ['1']
        assert pretty_number([0]) == ['0']
        assert pretty_number([-1]) == ['-1']
        assert pretty_number([1.0001]) == ['1']
        assert pretty_number([1.01]) == ['1.01']
        assert pretty_number([1e-5]) == ['1.000e-05']
        assert pretty_number([1e5]) == ['1.000e+05']
        assert pretty_number([3.3]) == ['3.3']
        assert pretty_number([1.]) == ['1']
        assert pretty_number([1.200]) == ['1.2']

    def test_large(self):
        # Regression test or a bug that caused trailing zeros in exponent to
        # be removed.
        assert pretty_number([1e9]) == ['1.000e+09']
        assert pretty_number([2e10]) == ['2.000e+10']
        assert pretty_number([3e11]) == ['3.000e+11']

    def test_list(self):
        assert pretty_number([1, 2, 3.3, 1e5]) == ['1', '2', '3.3',
                                                   '1.000e+05']


def test_unbroadcast():

    x = np.array([1, 2, 3])
    y = broadcast_to(x, (2, 4, 3))

    z = unbroadcast(y)
    assert z.shape == (1, 1, 3)
    np.testing.assert_allclose(z[0, 0], x)


@pytest.mark.parametrize(('chunk_shape', 'n_max'),
                         [(None, 121), (None, 100000), (None, 5), ((3, 1, 2, 4, 1, 2), None)])
def test_iterate_chunks(chunk_shape, n_max):

    array = np.zeros((6, 3, 4, 5, 1, 8))

    for slices in iterate_chunks(array.shape, chunk_shape=chunk_shape, n_max=n_max):

        print(slices)

        # Make sure empty slices aren't returned
        assert array[slices].size > 0

        # Increment all values in slice by 1
        array[slices] += 1

    assert_equal(array, 1)


def test_iterate_chunks_invalid():

    array = np.zeros((6, 3, 4, 5, 1, 8))

    with pytest.raises(ValueError) as exc:
        next(iterate_chunks(array.shape, chunk_shape=(6, 2, 1)))
    assert exc.value.args[0] == 'chunk_shape should have the same length as shape'

    with pytest.raises(ValueError) as exc:
        next(iterate_chunks(array.shape, chunk_shape=(6, 2, 1, 5, 2, 8)))
    assert exc.value.args[0] == 'chunk_shape should fit within shape'


FUNCTIONS = [nanmean, nanmedian, nanmin, nanmax, nansum]
AXIS = [None, 0, 2, 3, (0, 1), (2, 3), (0, 1, 2), (0, 1, 2, 3)]
ARRAY = np.random.random((4, 5, 2, 7))
ARRAY[ARRAY < 0.1] = np.nan


@pytest.mark.parametrize(('function', 'axis'), product(FUNCTIONS, AXIS))
def test_nanfunctions(function, axis):
    name = function.__name__
    np_func = getattr(np, name)
    assert_allclose(function(ARRAY, axis=axis), np_func(ARRAY, axis=axis))


SLICE_CASES = [
    (slice(None), slice(5), 10),
    (slice(None), slice(1, 5), 10),
    (slice(None), slice(1, 5, 2), 10),
    (slice(1, 8), slice(1, 5), 10),
    (slice(1, 8), slice(1, 5, 2), 10),
    (slice(1, 9, 2), slice(2, 5), 10),
    (slice(1, 9, 2), slice(2, 6), 10),
    (slice(1, 20, 2), slice(3, 18, 2), 20),
    (slice(1, 20, 2), slice(4, 18, 2), 20),
    (slice(1, 20, 2), slice(4, 18, 3), 20),
    (slice(1, None), slice(None, None, 2), 2),
    (slice(2), slice(None), 3)]


@pytest.mark.parametrize(('slice1', 'slice2', 'length'), SLICE_CASES)
def test_combine_slices(slice1, slice2, length):

    # Rather than hard-code the expected result, we can directly check that
    # the resulting slice gives the same indices if applied after the first
    # compared to a manual check

    indices1 = list(range(*slice1.indices(length)))
    indices2 = list(range(*slice2.indices(length)))

    expected = [indices1.index(idx) for idx in indices2 if idx in indices1]

    actual = list(range(*combine_slices(slice1, slice2, length).indices(length)))

    assert actual == expected


if HYPOTHESIS_INSTALLED:

    from hypothesis import given, settings
    from hypothesis.strategies import none, integers

    @given(beg1=none() | integers(-100, 100),
           end1=none() | integers(-100, 100),
           stp1=none() | integers(1, 100),
           beg2=none() | integers(-100, 100),
           end2=none() | integers(-100, 100),
           stp2=none() | integers(1, 100),
           length=integers(0, 100))
    @settings(max_examples=10000, derandomize=True)
    def test_combine_slices_hypot(beg1, end1, stp1, beg2, end2, stp2, length):

        slice1 = slice(beg1, end1, stp1)
        slice2 = slice(beg2, end2, stp2)

        # Rather than hard-code the expected result, we can directly check that
        # the resulting slice gives the same indices if applied after the first
        # compared to a manual check

        indices1 = list(range(*slice1.indices(length)))
        indices2 = list(range(*slice2.indices(length)))

        expected = [indices1.index(idx) for idx in indices2 if idx in indices1]

        actual = list(range(*combine_slices(slice1, slice2, length).indices(length)))

        assert actual == expected


def test_format_minimal():

    # TODO: in future could consider detecting integer cases
    fmt, strings = format_minimal([133, 1444, 3300])
    assert fmt == "{:.1f}"
    assert strings == ['133.0', '1444.0', '3300.0']

    # TODO: in future could consider detecting if all intervals are integers
    fmt, strings = format_minimal([133., 1444., 3300.])
    assert fmt == "{:.1f}"
    assert strings == ['133.0', '1444.0', '3300.0']

    fmt, strings = format_minimal([3, 4.3, 4.4, 5])
    assert fmt == "{:.1f}"
    assert strings == ['3.0', '4.3', '4.4', '5.0']

    fmt, strings = format_minimal([3, 4.325, 4.326, 5])
    assert fmt == "{:.3f}"
    assert strings == ['3.000', '4.325', '4.326', '5.000']

    fmt, strings = format_minimal([-3, 0., 0.993e-4, 5])
    assert fmt == "{:.4f}"
    assert strings == ['-3.0000', '0.0000', '0.0001', '5.0000']

    fmt, strings = format_minimal([-3, 0., 0.993e-8, 5])
    assert fmt == "{:.1e}"
    assert strings == ['-3.0e+00', '0.0e+00', '9.9e-09', '5.0e+00']
