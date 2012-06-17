import unittest

from mock import MagicMock

import glue
from glue.data import ComponentID, Component, Data


class TestData(unittest.TestCase):
    def setUp(self):
        self.data = Data(label="Test Data")
        comp = MagicMock()
        comp.data.shape = (2,3)
        comp.shape = (2,3)
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

    def test_add_component_with_id(self):
        cid = glue.data.ComponentID("test")
        comp = MagicMock()
        comp.shape = (2,3)
        comp.units = None
        cid2 = self.data.add_component(comp, cid)
        self.assertIs(cid2, cid)

    def test_add_component_incompatible_shape(self):
        comp = MagicMock()
        comp.data.shape = (3,2)
        self.assertRaises(TypeError, self.data.add_component,
                          comp, "junk label")

    def test_component_ids(self):
        cid = self.data.component_ids()
        self.assertIn(self.comp_id, cid)

    def test_new_subset(self):
        sub = self.data.new_subset()
        self.assertIn(sub, self.data.subsets)

    def test_data_created_with_edit_subset(self):
        self.assertEquals(len(self.data.subsets), 1)

    def test_register(self):
        hub = MagicMock(spec_set = glue.Hub)
        not_hub = MagicMock()
        self.data.register_to_hub(hub)
        self.assertIs(hub, self.data.hub)
        self.assertRaises(TypeError, self.data.register_to_hub, not_hub)

    def test_broadcast(self):
        hub = MagicMock(spec_set = glue.Hub)

        # make sure broadcasting with no hub is ok
        self.data.broadcast()

        # make sure broadcast with hub gets relayed
        self.data.register_to_hub(hub)
        self.data.broadcast()
        self.assertEquals(hub.broadcast.call_count, 1)

    def test_clone_subset(self):
        sub1 = self.data.new_subset()
        sub2 = self.data.create_subset_from_clone(sub1)
        self.assertIs(sub1.subset_state, sub2.subset_state)

    def test_double_hub_add(self):
        hub = MagicMock(spec_set = glue.Hub)
        hub2 = MagicMock(spec_set = glue.Hub)
        self.data.register_to_hub(hub)
        self.assertRaises(AttributeError, self.data.__setattr__, 'hub', hub2)

    def test_primary_components(self):
        compid = glue.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.component_link.ComponentLink)
        comp = glue.data.DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.primary_components
        self.assertIn(self.comp_id, pricomps)
        self.assertNotIn(compid, pricomps)

    def test_add_component_link(self):
        compid = glue.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.component_link.ComponentLink)
        cid = glue.data.ComponentID("new id")
        link.get_to_id.return_value = cid

        self.data.add_component_link(link)
        self.assertIn(cid, self.data.derived_components)

    def test_derived_components(self):
        compid = glue.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.component_link.ComponentLink)
        comp = glue.data.DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.derived_components
        self.assertNotIn(self.comp_id, pricomps)
        self.assertIn(compid, pricomps)


    def test_add_derived_component(self):
        compid = glue.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.component_link.ComponentLink)
        comp = glue.data.DerivedComponent(self.data, link)
        self.data.add_component(comp, compid)

        result = self.data[compid]
        link.compute.assert_called_once_with(self.data)

    def test_find_component_id(self):
        cid = self.data.find_component_id('Test Component')
        self.assertItemsEqual(cid, [self.comp_id])

    def test_add_subset(self):
        s = MagicMock(spec_set = glue.Subset)
        self.data.add_subset(s)
        self.assertIn(s, self.data.subsets)

    def test_add_subset_with_hub(self):
        s = MagicMock(spec_set = glue.Subset)
        hub = MagicMock(spec_set = glue.Hub)
        self.data.register_to_hub(hub)

        self.data.add_subset(s)
        self.assertIn(s, self.data.subsets)
        self.assertEquals(hub.broadcast.call_count, 1)

    def test_remove_subset(self):
        s = MagicMock(spec_set = glue.Subset)
        self.data.add_subset(s)
        self.data.remove_subset(s)
        self.assertNotIn(s, self.data.subsets)

    def test_remove_subset_with_hub(self):
        s = MagicMock(spec_set = glue.Subset)
        hub = MagicMock(spec_set = glue.Hub)

        self.data.register_to_hub(hub)
        self.data.add_subset(s)
        self.data.remove_subset(s)

        self.assertNotIn(s, self.data.subsets)
        self.assertEquals(hub.broadcast.call_count, 2)

    def test_get_component(self):
        self.assertIs(self.data.get_component(self.comp_id), self.comp)

    def test_get_item(self):
        self.assertIs(self.data[self.comp_id], self.comp.data)


if __name__ == "__main__":
    unittest.main()