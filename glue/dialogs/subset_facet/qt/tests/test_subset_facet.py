from unittest.mock import patch
from matplotlib import cm

from glue.core import Data, DataCollection

from ..subset_facet import SubsetFacetDialog


patched_facet = patch('glue.dialogs.subset_facet.qt.subset_facet.facet_subsets')


class TestSubsetFacet(object):

    def setup_method(self, method):
        d = Data(x=[1, 2, 3])
        dc = DataCollection([d])
        self.collect = dc
        self.s = dc.new_subset_group()

    def test_limits(self):
        s = SubsetFacetDialog(self.collect)
        s.state.data = self.collect[0]
        s.state.att = self.collect[0].id['x']
        assert s.state.v_min == 1
        assert s.state.v_max == 3

    def test_get_set_cmap(self):
        s = SubsetFacetDialog(self.collect)
        assert s.state.cmap is cm.RdYlBu

    def test_apply(self):
        with patched_facet as p:
            s = SubsetFacetDialog(self.collect)
            s.state.data = self.collect[0]
            s.state.att = self.collect[0].id['x']
            s._apply()
            p.assert_called_once_with(self.collect, s.state.att,
                                      lo=1, hi=3,
                                      steps=5, log=False,
                                      cmap=cm.RdYlBu)
