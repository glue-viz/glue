# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import numpy as np

from ..util import (file_format, point_contour, view_shape, facet_subsets,
                    colorize_subsets, coerce_numeric, as_variable_name)


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
    assert view_shape((10, 10), ([1, 2, 3], [2, 3, 4])) == (3,)


class TestFacetSubsets(object):

    def setup_method(self, method):
        from glue.core import Data, DataCollection
        self.data = Data(label='data', x=[1, 2, 3, 4, 5, 6, 7])
        self.collect = DataCollection([self.data])

    def test_facet_fully_specified(self):
        grps = facet_subsets(self.collect, self.data.id['x'],
                             lo=3, hi=6, steps=3)
        assert len(grps) == 3
        np.testing.assert_array_equal(grps[0].subsets[0].to_mask(),
                                      [False, False, True,
                                       False, False, False, False])
        np.testing.assert_array_equal(grps[1].subsets[0].to_mask(),
                                      [False, False, False,
                                       True, False, False, False])
        np.testing.assert_array_equal(grps[2].subsets[0].to_mask(),
                                      [False, False, False,
                                       False, True, False, False])

    def test_default_lo_value(self):
        grps = facet_subsets(self.collect, self.data.id['x'],
                             hi=7, steps=2)
        assert len(grps) == 2
        np.testing.assert_array_equal(grps[0].subsets[0].to_mask(),
                                      [True, True, True, False,
                                       False, False, False])
        np.testing.assert_array_equal(grps[1].subsets[0].to_mask(),
                                      [False, False, False, True,
                                       True, True, False])

    def test_default_hi_value(self):
        grps = facet_subsets(self.collect, self.data.id['x'],
                             lo=3, steps=2)
        assert len(grps) == 2
        np.testing.assert_array_equal(grps[0].subsets[0].to_mask(),
                                      [False, False, True, True, False,
                                       False, False])
        np.testing.assert_array_equal(grps[1].subsets[0].to_mask(),
                                      [False, False, False, False, True,
                                       True, False])

    def test_default_steps(self):
        grps = facet_subsets(self.collect, self.data.id['x'])
        assert len(grps) == 5

    def test_label(self):
        grps = facet_subsets(self.collect, self.data.id['x'])
        lbls = ['1.0<=x<2.2', '2.2<=x<3.4', '3.4<=x<4.6', '4.6<=x<5.8',
                '5.8<=x<7.0', None]
        for s, lbl in zip(grps, lbls):
            assert s.label == lbl

        grps = facet_subsets(self.collect, self.data.id['x'], prefix='test_')
        for i, s in enumerate(grps, start=1):
            assert s.label.startswith('test_')

    def test_facet_reversed(self):
        grps = facet_subsets(self.collect, self.data.id['x'],
                             lo=3, hi=1, steps=2)
        assert len(grps) == 2
        # ranges should be (2, 3] and (1, 2]
        np.testing.assert_array_equal(grps[0].subsets[0].to_mask(),
                                      [False, False, True, False, False,
                                       False, False])
        np.testing.assert_array_equal(grps[1].subsets[0].to_mask(),
                                      [False, True, False, False, False,
                                       False, False])


def test_colorize_subsets():
    from glue.core import Data, DataCollection
    from matplotlib.cm import gray

    data = Data(label='test', x=[1, 2, 3])
    dc = DataCollection(data)
    grps = facet_subsets(dc, data.id['x'], steps=2)
    colorize_subsets(grps, gray)

    assert grps[0].style.color == '#000000'
    assert grps[1].style.color == '#ffffff'


def test_colorize_subsets_clip():
    from glue.core import Data, DataCollection
    from matplotlib.cm import gray

    data = Data(label='test', x=[1, 2, 3])
    grps = facet_subsets(DataCollection(data), data.id['x'], steps=2)

    colorize_subsets(grps, gray, hi=0.5)
    assert grps[0].style.color == '#000000'
    assert grps[1].style.color == '#808080'

    colorize_subsets(grps, gray, lo=0.5)
    assert grps[0].style.color == '#808080'
    assert grps[1].style.color == '#ffffff'


def test_coerce_numeric():

    x = np.array(['1', '2', '3.14', '4'], dtype=str)

    np.testing.assert_array_equal(coerce_numeric(x),
                                  [1, 2, 3.14, 4])

    x = np.array([1, 2, 3])

    assert x is coerce_numeric(x)


def test_as_variable_name():
    def check(input, expected):
        assert as_variable_name(input) == expected

    tests = [('x', 'x'),
             ('x2', 'x2'),
             ('2x', '_2x'),
             ('x!', 'x_'),
             ('x y z', 'x_y_z'),
             ('_XY', '_XY')
             ]
    for input, expected in tests:
        yield check, input, expected
