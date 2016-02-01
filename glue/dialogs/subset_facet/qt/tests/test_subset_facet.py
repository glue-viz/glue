from __future__ import absolute_import, division, print_function

from mock import patch
from matplotlib import cm

from glue.core import Data, DataCollection

from ..subset_facet import SubsetFacet


patched_facet = patch('glue.dialogs.subset_facet.qt.subset_facet.facet_subsets')


class TestSubsetFacet(object):

    def setup_method(self, method):
        d = Data(x=[1, 2, 3])
        dc = DataCollection([d])
        self.collect = dc
        self.s = dc.new_subset_group()

    def test_limits(self):
        s = SubsetFacet(self.collect)
        s.data = self.collect[0]
        s.component = self.collect[0].id['x']
        assert s.vmin == 1
        assert s.vmax == 3

    def test_get_set_cmap(self):
        s = SubsetFacet(self.collect)
        assert s.cmap is cm.cool

    def test_apply(self):
        with patched_facet as p:
            s = SubsetFacet(self.collect)
            s.data = self.collect[0]
            s.component = self.collect[0].id['x']
            s._apply()
            p.assert_called_once_with(self.collect, s.component,
                                      lo=1, hi=3,
                                      steps=5, log=False)
