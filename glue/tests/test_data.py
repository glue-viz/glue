import unittest

from mock import MagicMock

import glue
from glue.data import ComponentID, Component, Data


class TestData(unittest.TestCase):
    def setUp(self):
        self.data = Data(label="Test Data")
        comp = MagicMock()
        comp.data.shape = (2,3)
        comp.units = None
        self.comp = comp
        self.comp_id = self.data.add_component(comp, 'Test Component')

    def test_shape_empty(self):
        d = Data()
        self.assertEquals(d.shape, None)

    def test_ndim_empty(self):
        d = Data()
        self.assertEquals(d.ndim, 0)

    def test_shape(self):
        self.assertEquals(self.data.shape, (2,3))

    def test_ndim(self):
        self.assertEquals(self.data.ndim, 2)

    def test_label(self):
        d = Data()
        self.assertEquals(d.label, None)
        self.assertEquals(self.data.label, "Test Data")

    def test_add_component_incompatible_shape(self):
        comp = MagicMock()
        comp.data.shape = (3,2)
        self.assertRaises(TypeError, self.data.add_component,
                          comp, "junk label")


if __name__ == "__main__":
    unittest.main()