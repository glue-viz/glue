# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from numpy import array

from glue.core.data import ComponentID
from glue import core

from ..component_selector import ComponentSelector


def data_collection():
    d = core.data.Data(label='test data')
    c1 = core.data.Component(array([1, 2, 3]))
    c2 = core.data.Component(array([1, 2, 3]))
    dc = core.data_collection.DataCollection()
    dc.append(d)
    d.add_component(c1, 'test1')
    d.add_component(c2, 'test2')
    dc.append(core.data.Data(label='test 2'))
    return dc


class TestComponentSelector(object):

    def setup_method(self, method):
        self.comp = ComponentSelector()
        self.data = data_collection()
        self.comp.setup(self.data)

    def test_component(self):
        self.comp.set_current_row(1)
        c = self.comp.component
        assert isinstance(c, ComponentID)

    def test_data(self):
        self.comp.set_data_row(0)
        assert self.comp.data is self.data[0]

        self.comp.set_data_row(1)
        assert self.comp.data is self.data[1]
