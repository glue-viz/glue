# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib.cm import gray

from .. import Data, DataCollection
from ..util import facet_subsets, colorize_subsets


class TestRelim(object):
    pass


class TestFacetSubsets(object):

    def setup_method(self, method):
        from .. import Data, DataCollection
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

    data = Data(label='test', x=[1, 2, 3])
    dc = DataCollection(data)
    grps = facet_subsets(dc, data.id['x'], steps=2)
    colorize_subsets(grps, gray)

    assert grps[0].style.color == '#000000'
    assert grps[1].style.color == '#ffffff'


def test_colorize_subsets_clip():

    data = Data(label='test', x=[1, 2, 3])
    grps = facet_subsets(DataCollection(data), data.id['x'], steps=2)

    colorize_subsets(grps, gray, hi=0.5)
    assert grps[0].style.color == '#000000'
    assert grps[1].style.color == '#808080'

    colorize_subsets(grps, gray, lo=0.5)
    assert grps[0].style.color == '#808080'
    assert grps[1].style.color == '#ffffff'
