import os

import pytest

from numpy.testing import assert_allclose

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Galactic, ICRS
from astropy.wcs import WCS

from ..geometry import Path

HEADER_FILE = os.path.join(os.path.dirname(__file__), 'data', 'w51.hdr')
HEADER = fits.Header.fromtextfile(HEADER_FILE)


class TestPixel(object):

    def setup_method(self, method):

        self.pixel = [(0., 0.), (50., 50.)]

    def test_get_xy(self):
        path = Path(self.pixel)
        assert_allclose(path.get_xy(), [(0., 0.), (50., 50.)])

    def test_add_point(self):
        path = Path(self.pixel)
        path.add_point((75., 60.))
        assert_allclose(path.get_xy(), [(0., 0.), (50., 50.), (75., 60.)])

    def test_add_point_invalid(self):
        path = Path(self.pixel)
        with pytest.raises(TypeError) as exc:
            path.add_point(Galactic(49.5 * u.deg, -0.5 * u.deg))
        assert exc.value.args[0] == ("Path is defined as a list of pixel "
                                     "coordinates, so `xy_or_coord` "
                                     "should be a tuple of `(x,y)` "
                                     "pixel coordinates.")


class TestCoord(object):

    def setup_method(self, method):

        self.gal = Galactic([49.4, 49.3, 49.] * u.deg,
                            [-0.4, -0.1, 0.] * u.deg)

        self.icrs = ICRS([290.77261, 290.70523, 290.71415] * u.deg,
                         [14.638866, 14.30515, 14.012249] * u.deg)

    def test_get_xy_noargs(self):

        path = Path(self.gal)

        with pytest.raises(ValueError) as exc:
            path.get_xy()
        assert exc.value.args[0] == '`wcs` is needed in order to compute the pixel coordinates'

    def test_add_point(self):
        path = Path(self.gal)
        with pytest.raises(NotImplementedError):
            path.add_point(Galactic(49.5 * u.deg, -0.5 * u.deg))

    def test_add_point_invalid(self):
        path = Path(self.gal)
        with pytest.raises(TypeError) as exc:
            path.add_point((40., 50.))
        assert exc.value.args[0] == ("Path is defined in world coordinates, "
                                     "so `xy_or_coord` should be an Astropy "
                                     "coordinate object.")

    def test_pixel_same_frame(self):

        # Specify coordinates in the same frame as the WCS

        path = Path(self.gal)

        wcs = WCS(HEADER)

        assert_allclose(path.get_xy(wcs=wcs),
                        [(107.29272, 71.51288),
                         (131.29272, 143.51288),
                         (203.29272, 167.51288)])

    def test_pixel_diff_frame(self):

        # Specify coordinates in a different frame from the WCS

        path = Path(self.icrs)

        wcs = WCS(HEADER)

        assert_allclose(path.get_xy(wcs=wcs),
                        [(76, 122),
                         (154, 98),
                         (215, 63)], rtol=1.e-3)


def test_sample_points_edges():

    d, x, y = Path(zip([0., 0.], [0., 3.4])).sample_points_edges(spacing=1.0)

    assert_allclose(d, [0., 1., 2., 3.])
    assert_allclose(x, [0., 0., 0., 0.])
    assert_allclose(y, [0., 1., 2., 3.])

    d, x, y = Path(zip([0., 0., 0.], [0., 2.0, 3.4])).sample_points_edges(spacing=1.0)

    assert_allclose(d, [0., 1., 2., 3.])
    assert_allclose(x, [0., 0., 0., 0.])
    assert_allclose(y, [0., 1., 2., 3.])

    d, x, y = Path(zip([0., 0., 0.], [0., 2.0, 3.4])).sample_points_edges(spacing=2.0)

    assert_allclose(d, [0., 2.])
    assert_allclose(x, [0., 0.])
    assert_allclose(y, [0., 2.])

    d, x, y = Path(zip([0., 3.4], [0., 0.])).sample_points_edges(spacing=1.0)

    assert_allclose(d, [0., 1., 2., 3.])
    assert_allclose(x, [0., 1., 2., 3.])
    assert_allclose(y, [0., 0., 0., 0.])

    d, x, y = Path(zip([0., 2.0, 3.4], [0., 0., 0.])).sample_points_edges(spacing=1.0)

    assert_allclose(d, [0., 1., 2., 3.])
    assert_allclose(x, [0., 1., 2., 3.])
    assert_allclose(y, [0., 0., 0., 0.])

    d, x, y = Path(zip([0., 2.0, 3.4], [0., 0., 0.])).sample_points_edges(spacing=2.0)

    assert_allclose(d, [0., 2.])
    assert_allclose(x, [0., 2.])
    assert_allclose(y, [0., 0.])


def test_sample_points():

    x, y = Path(zip([0., 3], [0., 4])).sample_points(spacing=1.0)

    assert_allclose(x, [0.3, 0.9, 1.5, 2.1, 2.7])
    assert_allclose(y, [0.4, 1.2, 2.0, 2.8, 3.6])


def test_sample_points_invalid_spacing():

    with pytest.raises(ValueError) as exc:
        x, y = Path(zip([0., 3], [0., 4])).sample_points(spacing=10.0)
    assert exc.value.args[0] == "Path is shorter than spacing"


def test_sample_polygons():

    path = Path(zip([0., 3., -1., 3, -3], [0., 4, 2, -4, 0]), width=1.)

    polygons = path.sample_polygons(spacing=1.)

    # TODO: check output below, then copy reference values to avoid regression

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Polygon
    #
    # fig = plt.figure(figsize=(5,5))
    # ax = fig.add_subplot(1,1,1)
    # for poly in polygons:
    #     ax.add_patch(Polygon(zip(poly.x, poly.y),
    #                          ec='green', fc='none'))
    # xy = path.get_xy()
    # x, y = zip(*xy)
    # ax.plot(x, y, 'o')
    # ax.set_xlim(-5., 5)
    # ax.set_ylim(-5., 5)
    # fig.savefig('test_poly.png')

