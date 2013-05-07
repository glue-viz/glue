#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import numpy as np

from ..util import (file_format, point_contour, view_shape, facet_subsets,
                    colorize_subsets)


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


class TestFacetSubsets(object):
    def setup_method(self, method):
        from glue.core import Data
        self.data = Data(label='data', x=[1, 2, 3, 4, 5, 6, 7])

    def test_facet_fully_specified(self):
        subsets = facet_subsets(self.data, self.data.id['x'],
                                lo=3, hi=6, steps=3)
        assert len(subsets) == 3
        np.testing.assert_array_equal(subsets[0].to_mask(),
                                      [False, False, True,
                                       False, False, False, False])
        np.testing.assert_array_equal(subsets[1].to_mask(),
                                      [False, False, False,
                                       True, False, False, False])
        np.testing.assert_array_equal(subsets[2].to_mask(),
                                      [False, False, False,
                                       False, True, False, False])

    def test_default_lo_value(self):
        subsets = facet_subsets(self.data, self.data.id['x'],
                                hi=7, steps=2)
        assert len(subsets) == 2
        np.testing.assert_array_equal(subsets[0].to_mask(),
                                      [True, True, True, False,
                                       False, False, False])
        np.testing.assert_array_equal(subsets[1].to_mask(),
                                      [False, False, False, True,
                                       True, True, False])

    def test_default_hi_value(self):
        subsets = facet_subsets(self.data, self.data.id['x'],
                                lo=3, steps=2)
        assert len(subsets) == 2
        np.testing.assert_array_equal(subsets[0].to_mask(),
                                      [False, False, True, True, False,
                                       False, False])
        np.testing.assert_array_equal(subsets[1].to_mask(),
                                      [False, False, False, False, True,
                                       True, False])

    def test_default_steps(self):
        subsets = facet_subsets(self.data, self.data.id['x'])
        assert len(subsets) == 5

    def test_prefix(self):
        subsets = facet_subsets(self.data, self.data.id['x'], prefix='test')
        for i, s in enumerate(subsets, start=1):
            assert s.label == "test_%i" % i

        subsets = facet_subsets(self.data, self.data.id['x'])
        for i, s in enumerate(subsets, start=1):
            assert s.label.startswith('data')


def test_colorize_subsets():
    from glue.core import Data
    from matplotlib.cm import gray

    data = Data(label='test', x=[1, 2, 3])
    subsets = facet_subsets(data, data.id['x'], steps=2)
    colorize_subsets(subsets, gray)

    assert subsets[0].style.color == '#000000'
    assert subsets[1].style.color == '#ffffff'


def test_colorize_subsets_clip():
    from glue.core import Data
    from matplotlib.cm import gray

    data = Data(label='test', x=[1, 2, 3])
    subsets = facet_subsets(data, data.id['x'], steps=2)

    colorize_subsets(subsets, gray, hi=0.5)
    assert subsets[0].style.color == '#000000'
    assert subsets[1].style.color == '#808080'

    colorize_subsets(subsets, gray, lo=0.5)
    assert subsets[0].style.color == '#808080'
    assert subsets[1].style.color == '#ffffff'


def test_coerce_numeric():
    from ..util import coerce_numeric

    x = np.array(['1', '2', '3.14', '4'])

    np.testing.assert_array_equal(coerce_numeric(x),
                                  [1, 2, 3.14, 4])

    x = np.array([1, 2, 3])

    assert x is coerce_numeric(x)
