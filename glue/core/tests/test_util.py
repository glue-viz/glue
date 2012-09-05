#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import numpy as np

from ..util import file_format, point_contour, view_shape


class TestRelim(object):
    pass


class TestFileFormat(object):

    def test_gz(self):
        fmt = file_format('test.tar.gz')
        assert fmt == 'tar'

    def test_normal(self):
        fmt = file_format('test.data')
        assert fmt == 'data'

    def test_underscores(self):
        fmt = file_format('test_file.fits_file')
        assert fmt == 'fits_file'

    def test_multidot(self):
        fmt = file_format('test.a.b.c')
        assert fmt == 'c'

    def test_nodot(self):
        fmt = file_format('test')
        assert fmt == ''


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


def test_view_shape():
    assert view_shape((10, 10), np.s_[:]) == (10, 10)
    assert view_shape((10, 10, 10), np.s_[:]) == (10, 10, 10)
    assert view_shape((10, 10), np.s_[:, 1]) == (10,)
    assert view_shape((10, 10), np.s_[2:3, 2:3]) == (1, 1)
    assert view_shape((10, 10), None) == (10, 10)
