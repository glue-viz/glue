import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ...tests.helpers import requires_scipy

from ..matplotlib import point_contour, fast_limits, all_artists, new_artists, remove_artists, view_cascade, get_extent


@requires_scipy
class TestPointContour(object):

    def test(self):
        data = np.array([[0, 0, 0, 0],
                         [0, 2, 3, 0],
                         [0, 4, 2, 0],
                         [0, 0, 0, 0]])
        xy = point_contour(2, 2, data)
        x = np.array([2., 2. + 1. / 3., 2., 2., 1, .5, 1, 1, 2])
        y = np.array([2. / 3., 1., 2., 2., 2.5, 2., 1., 1., 2. / 3])

        np.testing.assert_array_almost_equal(xy[:, 0], x)
        np.testing.assert_array_almost_equal(xy[:, 1], y)


def test_fast_limits_nans():
    x = np.zeros((10, 10)) * np.nan
    assert_allclose(fast_limits(x, 0, 1), [0, 1])


def test_single_value():
    x = np.array([1])
    assert_allclose(fast_limits(x, 5., 95.), [1, 1])


def test_artist_functions():

    c1 = Circle((0, 0), radius=1)
    c2 = Circle((1, 0), radius=1)
    c3 = Circle((2, 0), radius=1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.add_patch(c1)
    ax.add_patch(c2)
    assert all_artists(fig) == set([c1, c2])

    ax.add_patch(c3)
    assert new_artists(fig, set([c1, c2])) == set([c3])

    remove_artists([c2])
    assert all_artists(fig) == set([c1, c3])

    # check that it can deal with being passed the same artist twice
    remove_artists([c1, c1])
    assert all_artists(fig) == set([c3])


def test_get_extent():
    assert get_extent((slice(0, 5, 1), slice(0, 10, 2))) == (0, 10, 0, 5)
    assert get_extent((slice(0, 5, 1), slice(0, 10, 2)), transpose=True) == (0, 5, 0, 10)


def test_view_cascade():
    data = np.zeros((100, 100))

    v2, view = view_cascade(data, (slice(0, 5, 1), slice(0, 5, 1)))
    assert v2 == ((slice(0, 100, 20), slice(0, 100, 20)))
    assert view == (slice(0, 5, 1), slice(0, 5, 1))

    v2, view = view_cascade(data, (3, slice(0, 5, 1)))
    assert v2 == ((3, slice(0, 100, 20)))
    assert view == (3, slice(0, 5, 1))
