from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from numpy.testing import assert_allclose

from .. import Data
from ..aggregate import Aggregate


class TestFunctions(object):

    def setup_method(self, method):
        self.d = Data(a=np.random.random((3, 3, 3)) - 0.5)
        self.agg = Aggregate(self.d, 'a', 0, (0, 'y', 'x'), (0, 3))

    def test_max(self):
        actual = self.agg.max()
        expected = self.d['a'].max(axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_mean(self):
        actual = self.agg.mean()
        expected = self.d['a'].mean(axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_median(self):
        actual = self.agg.median()
        expected = np.median(self.d['a'], axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_argmax(self):
        actual = self.agg.argmax()
        expected = np.nanargmax(self.d['a'], axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_argmin(self):
        actual = self.agg.argmin()
        expected = np.nanargmin(self.d['a'], axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_mom1(self):
        actual = self.agg.mom1()
        a = np.maximum(self.d['a'], 0)
        z = self.d[self.d.get_world_component_id(0)]
        expected = (a * z).sum(axis=0) / a.sum(axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_mom2(self):
        # this is a different implementation, as a sanity check
        actual = self.agg.mom2()

        # negative values clipped at 0 for weight calculation
        a = np.maximum(self.d['a'], 0)
        z = self.d[self.d.get_world_component_id(0)]
        a = a / a.sum(axis=0)
        mom1 = self.agg.mom1()

        expected = np.sqrt((a * (z - mom1) ** 2).sum(axis=0))
        actual = self.agg.mom2()

        np.testing.assert_array_almost_equal(expected, actual)


class TestSliceDescriptions(object):

    """Look at various slice orientations and limits"""

    def test_cube(self):
        d = Data(a=np.random.random((3, 3, 3)))

        actual = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3)).sum()
        expected = d['a'].sum(axis=0)

        np.testing.assert_array_equal(expected, actual)

    def test_cube_transpose(self):
        d = Data(a=np.random.random((3, 3, 3)))

        actual = Aggregate(d, 'a', 0, (0, 'x', 'y'), (0, 3)).sum()
        expected = d['a'].sum(axis=0).T

        np.testing.assert_array_equal(expected, actual)

    def test_cube_axis1(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 1, ('x', 0, 'y'), (0, 3)).sum()
        expected = d['a'].sum(axis=1).T

        np.testing.assert_array_equal(expected, actual)

    def test_cube_zlim(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 0, (0, 'x', 'y'), (0, 2)).sum()
        expected = d['a'][0:2].sum(axis=0).T

        np.testing.assert_array_equal(expected, actual)

    def test_4cube(self):
        d = Data(a=np.random.random((3, 3, 3, 3)))
        actual = Aggregate(d, 'a', 2, ('x', 2, 0, 'y'), (0, 3)).sum()
        expected = d['a'][:, 2, :, :].sum(axis=1).T

        np.testing.assert_array_equal(expected, actual)

    @pytest.mark.parametrize('func', Aggregate.all_operators())
    def test_empty_slices(self, func):
        d = Data(a=np.random.random((3, 4, 5)))
        a = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 0))
        actual = func(a)
        expected = np.zeros((4, 5)) * np.nan
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize('func', Aggregate.all_operators())
def test_transpose(func):
    d = Data(a=np.random.random((3, 3, 3)))
    a1 = Aggregate(d, 'a', 0, (0, 'x', 'y'), (0, 3))
    a2 = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3))
    np.testing.assert_array_equal(func(a1).T, func(a2))


@pytest.mark.parametrize('func',
                         (Aggregate.sum, Aggregate.mom1, Aggregate.mom2))
def test_nans_like_zeros(func):
    a = np.random.random((3, 3, 3))
    a[0] = np.nan
    d = Data(a=a)
    d2 = Data(a=np.nan_to_num(a))

    a1 = Aggregate(d, 'a', 0, (0, 'x', 'y'), (0, 3))
    a2 = Aggregate(d2, 'a', 0, (0, 'x', 'y'), (0, 3))
    np.testing.assert_array_equal(func(a1), func(a2))


def test_reverse_limits():
    a = np.random.random((3, 3, 3))
    d = Data(a=a)

    a = Aggregate(d, 'a', 0, (0, 'y', 'x'), (3, 0))
    b = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3))
    assert_allclose(a.sum(), b.sum())
