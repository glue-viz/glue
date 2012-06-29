import unittest

from mock import MagicMock

from glue.data import Component, ComponentID, DerivedComponent


class TestComponent(unittest.TestCase):

    def setUp(self):
        self.data = MagicMock()
        self.data.shape = [1,2]
        self.component = Component(self.data)

    def test_data(self):
        self.assertIs(self.component.data, self.data)

    def test_shape(self):
        self.assertIs(self.component.shape, self.data.shape)

    def test_ndim(self):
        self.assertIs(self.component.ndim, len(self.data.shape))

class TestComponentID(unittest.TestCase):
    def setUp(self):
        self.cid = ComponentID('test')

    def test_label(self):
        self.assertEquals(self.cid.label, 'test')

    def test_str(self):
        """ str should return """
        str(self.cid)

    def test_repr(self):
        """ str should return """
        repr(self.cid)

class TestDerivedComponent(unittest.TestCase):
    def setUp(self):
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
        self.assertEquals(self.cid.link, self.link)

if __name__ == "__main__":
    unittest.main()