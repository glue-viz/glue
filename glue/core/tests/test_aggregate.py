import numpy as np

from ..aggregate import Aggregate
from .. import Data


class TestAggregate(object):

    def test_sum_cube(self):
        d = Data(a=np.random.random((3, 3, 3)))

        actual = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3)).sum()
        expected = d['a'].sum(axis=0)

        np.testing.assert_array_equal(expected, actual)

    def test_sum_cube_transpose(self):
        d = Data(a=np.random.random((3, 3, 3)))

        actual = Aggregate(d, 'a', 0, (0, 'x', 'y'), (0, 3)).sum()
        expected = d['a'].sum(axis=0).T

        np.testing.assert_array_equal(expected, actual)

    def test_sum_cube_axis1(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 1, ('x', 0, 'y'), (0, 3)).sum()
        expected = d['a'].sum(axis=1).T

        np.testing.assert_array_equal(expected, actual)

    def test_sum_cube_zlim(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 0, (0, 'x', 'y'), (0, 2)).sum()
        expected = d['a'][0:2].sum(axis=0).T

        np.testing.assert_array_equal(expected, actual)

    def test_sum_4cube(self):
        d = Data(a=np.random.random((3, 3, 3, 3)))
        actual = Aggregate(d, 'a', 2, ('x', 2, 0, 'y'), (0, 3)).sum()
        expected = d['a'][:, 2, :, :].sum(axis=1).T

        np.testing.assert_array_equal(expected, actual)

    def test_max(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3)).max()
        expected = d['a'].max(axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_mean(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3)).mean()
        expected = d['a'].mean(axis=0)
        np.testing.assert_array_equal(expected, actual)

    def test_mom1(self):
        d = Data(a=np.random.random((3, 3, 3)))
        actual = Aggregate(d, 'a', 0, (0, 'y', 'x'), (0, 3)).mom1()

        a = d['a']
        z = d[d.get_world_component_id(0)]
        expected = (a * z).sum(axis=0) / a.sum(axis=0)
        np.testing.assert_array_equal(expected, actual)
