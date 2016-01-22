# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import numpy as np

from ..data import Data, Component


class TestDataRetrieval(object):

    def setup_method(self, method):

        data1 = Data()
        comp1 = Component(np.arange(5))
        id1 = data1.add_component(comp1, 'comp_1')
        comp2 = Component(np.arange(5) * 2)
        id2 = data1.add_component(comp2, 'comp_2')

        data2 = Data()
        comp3 = Component(np.arange(5) * 3)
        id3 = data2.add_component(comp3, 'comp_3')
        comp4 = Component(np.arange(5) * 4)
        id4 = data2.add_component(comp4, 'comp_4')

        self.data = [data1, data2]
        self.components = [comp1, comp2, comp3, comp4]
        self.component_ids = [id1, id2, id3, id4]

    def test_direct_get(self):
        assert self.data[0][self.component_ids[0]] is self.components[0].data
        assert self.data[0][self.component_ids[1]] is self.components[1].data
        assert self.data[1][self.component_ids[2]] is self.components[2].data
        assert self.data[1][self.component_ids[3]] is self.components[3].data
