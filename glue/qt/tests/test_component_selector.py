from PyQt4 import QtCore, QtGui
from numpy import array

from ..component_selector import ComponentSelector
from ... import core
from ...core.data import ComponentID

def setup_module(module):
    module.app = QtGui.QApplication([''])

def data_collection():
    d = core.data.Data(label='test data')
    c1 = core.data.Component(array([1,2,3]))
    c2 = core.data.Component(array([1,2,3]))
    dc = core.data_collection.DataCollection()
    dc.append(d)
    d.add_component(c1, 'test1')
    d.add_component(c2, 'test2')
    return dc

class TestComponentSelector(object):

    def setup_method(self, method):
        self.comp = ComponentSelector()
        self.data = data_collection()
        self.comp.setup(self.data)

    def test_component(self):
        self.comp.set_current_row(1)
        c = self.comp.component
        assert type(c) == ComponentID
