from mock import MagicMock

from ..data import Component, ComponentID, DerivedComponent


class TestComponent(object):

    def setup_method(self, method):
        self.data = MagicMock()
        self.data.shape = [1,2]
        self.component = Component(self.data)

    def test_data(self):
        assert self.component.data is self.data

    def test_shape(self):
        assert self.component.shape is self.data.shape

    def test_ndim(self):
        assert self.component.ndim is len(self.data.shape)

class TestComponentID(object):

    def setup_method(self, method):
        self.cid = ComponentID('test')

    def test_label(self):
        assert self.cid.label == 'test'

    def test_str(self):
        """ str should return """
        str(self.cid)

    def test_repr(self):
        """ str should return """
        repr(self.cid)

class TestDerivedComponent(object):

    def setup_method(self, method):
        data = MagicMock()
        link = MagicMock()
        self.cid = DerivedComponent(data, link)
        self.link = link
        self.data = data

    def test_data(self):
        """ data property should wrap to links compute method """
        self.cid.data
        self.link.compute.assert_called_once_with(self.data)

    def test_link(self):
        assert self.cid.link == self.link
