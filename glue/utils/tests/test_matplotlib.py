

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.artist import Artist
from numpy.testing import assert_allclose
from matplotlib.backends.backend_agg import FigureCanvasAgg

from glue.tests.helpers import requires_scipy, requires_skimage
from glue.utils.misc import DeferredMethod

from ..matplotlib import (point_contour, fast_limits, all_artists, new_artists,
                          remove_artists, view_cascade, get_extent, color2rgb,
                          defer_draw, freeze_margins, datetime64_to_mpl,
                          mpl_to_datetime64, color2hex)


@requires_scipy
@requires_skimage
def test_point_contour():
    data = np.array([[0, 0, 0, 0],
                     [0, 2, 3, 0],
                     [0, 4, 2, 0],
                     [0, 0, 0, 0]])
    xy = point_contour(2, 2, data)
    x = np.array([2., 1., 0.5, 0.5, 1., 2., 2.5, 2.5, 2.])
    y = np.array([2.5, 2.5, 2., 1., 0.5, 0.5, 1., 2., 2.5])

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


def test_defer_draw():

    @defer_draw
    def draw_figure():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3], [4, 5, 6])
        fig.canvas.draw_idle()
        return 3.5

    result = draw_figure()

    # Make sure that the return value was passed through correctly
    assert result == 3.5


def test_defer_draw_exception(tmpdir):

    # Regression test for a bug that meant that if an exception happened during
    # drawing, the draw method was not restored correctly

    # Make sure we start off with a clean draw method
    assert not isinstance(FigureCanvasAgg.draw, DeferredMethod)

    class ProblematicArtist(Artist):
        def draw(self, *args, **kwargs):
            raise ValueError('You shall not pass!')

    @defer_draw
    def draw_figure():
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.add_artist(ProblematicArtist())
        fig.savefig(tmpdir.join('test.png').strpath)

    with pytest.raises(ValueError) as exc:
        result = draw_figure()
    assert exc.value.args[0] == 'You shall not pass!'

    # Make sure that draw is no longer a deferred method
    assert not isinstance(FigureCanvasAgg.draw, DeferredMethod)


@pytest.mark.parametrize(('color', 'rgb'),
                         (('red', (1, 0, 0)), ('green', (0, 0.5020, 0)), ('orange', (1., 0.6470, 0.))))
def test_color2rgb(color, rgb):
    assert_allclose(color2rgb(color), rgb, atol=0.001)


def test_color2hex():
    assert color2hex('red') == '#ff0000'


def test_freeze_margins():

    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(1, 1, 1)
    freeze_margins(ax, margins=[1, 1, 1, 1])

    # Note, we don't test the following since the defaults change depending
    # on the Matplotlib version

    # bbox = ax.get_position()
    # np.testing.assert_allclose(bbox.x0, 0.125)
    # np.testing.assert_allclose(bbox.y0, 0.1)
    # np.testing.assert_allclose(bbox.x1, 0.9)
    # np.testing.assert_allclose(bbox.y1, 0.9)

    fig.canvas.resize_event()

    bbox = ax.get_position()
    np.testing.assert_allclose(bbox.x0, 0.25)
    np.testing.assert_allclose(bbox.y0, 0.25)
    np.testing.assert_allclose(bbox.x1, 0.75)
    np.testing.assert_allclose(bbox.y1, 0.75)

    fig.set_size_inches(8, 8)
    fig.canvas.resize_event()

    bbox = ax.get_position()
    np.testing.assert_allclose(bbox.x0, 0.125)
    np.testing.assert_allclose(bbox.y0, 0.125)
    np.testing.assert_allclose(bbox.x1, 0.875)
    np.testing.assert_allclose(bbox.y1, 0.875)

    ax.resizer.margins = [0, 1, 2, 4]
    fig.canvas.resize_event()

    bbox = ax.get_position()
    np.testing.assert_allclose(bbox.x0, 0.)
    np.testing.assert_allclose(bbox.y0, 0.25)
    np.testing.assert_allclose(bbox.x1, 0.875)
    np.testing.assert_allclose(bbox.y1, 0.5)


def test_mpl_datetime64():
    # Make sure the mpl <-> datetime64 conversion round-trips
    mpl1 = 719313
    mpl2 = datetime64_to_mpl(mpl_to_datetime64(mpl1))
    assert mpl1 == mpl2
