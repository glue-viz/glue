import unittest

import numpy as np
from mock import MagicMock

import glue
from glue.core.data import ComponentID, Component, Data

class TestCoordinates(glue.core.coordinates.Coordinates):
    def pixel2world(self, *args):
        return [(i+2.) * a for i,a in enumerate(args)]

    def world2pixel(self, *args):
        return [a/(i+2.) for i,a in enumerate(args)]

class TestData(unittest.TestCase):
    def setUp(self):
        self.data = Data(label="Test Data")
        comp = MagicMock()
        comp.data.shape = (2,3)
        comp.shape = (2,3)
        comp.units = None
        self.comp = comp
        self.data.coords = TestCoordinates()
        self.comp_id = self.data.add_component(comp, 'Test Component')

    def test_shape_empty(self):
        d = Data()
        self.assertEquals(d.shape, ())

    def test_ndim_empty(self):
        d = Data()
        self.assertEquals(d.ndim, 0)

    def test_shape(self):
        self.assertEquals(self.data.shape, (2,3))

    def test_ndim(self):
        self.assertEquals(self.data.ndim, 2)

    def test_label(self):
        d = Data()
        self.assertEquals(d.label, '')
        self.assertEquals(self.data.label, "Test Data")

    def test_set_label(self):
        d = Data()
        d.label = 'test_set_label'
        self.assertEquals(d.label, 'test_set_label')

    def test_add_component_with_id(self):
        cid = glue.core.data.ComponentID("test")
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

    def test_get_getitem_incompatible_attribute(self):
        cid = ComponentID('bad')
        self.assertRaises(glue.core.exceptions.IncompatibleAttribute,
                          self.data.__getitem__, cid)

    def test_get_component_incompatible_attribute(self):
        cid = ComponentID('bad')
        self.assertRaises(glue.core.exceptions.IncompatibleAttribute,
                          self.data.get_component, cid)

    def test_component_ids(self):
        cid = self.data.component_ids()
        self.assertIn(self.comp_id, cid)

    def test_new_subset(self):
        sub = self.data.new_subset()
        self.assertIn(sub, self.data.subsets)

    def test_data_created_with_edit_subset(self):
        self.assertEquals(len(self.data.subsets), 1)

    def test_register(self):
        hub = MagicMock(spec_set = glue.core.hub.Hub)
        not_hub = MagicMock()
        self.data.register_to_hub(hub)
        self.assertIs(hub, self.data.hub)
        self.assertRaises(TypeError, self.data.register_to_hub, not_hub)

    def test_component_order(self):
        """Components should be returned in alphabetical order"""
        data = Data()
        comp = Component(np.array([1,2,3]))
        labels = 'asldfkjaAREGWoibasiwnsldkgajsldkgslkg'
        for label in labels:
            data.add_component(comp, label)
        ids = data.components
        labels = [cid.label.lower() for cid in ids]
        self.assertEqual(labels, sorted(labels))


    def test_broadcast(self):
        hub = MagicMock(spec_set = glue.core.hub.Hub)

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
        hub = MagicMock(spec_set = glue.core.hub.Hub)
        hub2 = MagicMock(spec_set = glue.core.hub.Hub)
        self.data.register_to_hub(hub)
        self.assertRaises(AttributeError, self.data.__setattr__, 'hub', hub2)

    def test_primary_components(self):
        compid = glue.core.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.core.component_link.ComponentLink)
        comp = glue.core.data.DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.primary_components
        self.assertIn(self.comp_id, pricomps)
        self.assertNotIn(compid, pricomps)

    def test_add_component_invalid_label(self):
        self.assertRaises(TypeError,
                          self.data.add_component, self.comp, label=5)

    def test_add_component_invalid_component(self):
        comp = glue.core.data.Component(np.array([1]))
        self.assertRaises(TypeError,
                          self.data.add_component, comp, label='bad')

    def test_add_component_link(self):
        compid = glue.core.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.core.component_link.ComponentLink)
        cid = glue.core.data.ComponentID("new id")
        link.get_to_id.return_value = cid

        self.data.add_component_link(link)
        self.assertIn(cid, self.data.derived_components)

    def test_derived_components(self):
        compid = glue.core.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.core.component_link.ComponentLink)
        comp = glue.core.data.DerivedComponent(self.data, link)

        self.data.add_component(comp, compid)

        pricomps = self.data.derived_components
        self.assertNotIn(self.comp_id, pricomps)
        self.assertIn(compid, pricomps)

    def test_str_empty(self):
        d = glue.core.data.Data()
        str(d)

    def test_str_(self):
        str(self.data)

    def test_add_derived_component(self):
        compid = glue.core.data.ComponentID('virtual')
        link = MagicMock(spec_set = glue.core.component_link.ComponentLink)
        comp = glue.core.data.DerivedComponent(self.data, link)
        self.data.add_component(comp, compid)

        result = self.data[compid]
        link.compute.assert_called_once_with(self.data)

    def test_find_component_id(self):
        cid = self.data.find_component_id('Test Component')
        self.assertItemsEqual(cid, [self.comp_id])

    def test_add_subset(self):
        s = MagicMock(spec_set = glue.core.subset.Subset)
        self.data.add_subset(s)
        self.assertIn(s, self.data.subsets)

    def test_add_subset_with_hub(self):
        s = MagicMock(spec_set = glue.core.subset.Subset)
        hub = MagicMock(spec_set = glue.core.hub.Hub)
        self.data.register_to_hub(hub)

        self.data.add_subset(s)
        self.assertIn(s, self.data.subsets)
        self.assertEquals(hub.broadcast.call_count, 1)

    def test_remove_component(self):
        self.data.remove_component(self.comp_id)
        self.assertNotIn(self.comp_id, self.data.components)


    def test_remove_subset(self):
        s = MagicMock(spec_set = glue.core.subset.Subset)
        self.data.add_subset(s)
        self.data.remove_subset(s)
        self.assertNotIn(s, self.data.subsets)

    def test_remove_subset_with_hub(self):
        s = MagicMock(spec_set = glue.core.subset.Subset)
        hub = MagicMock(spec_set = glue.core.hub.Hub)

        self.data.register_to_hub(hub)
        self.data.add_subset(s)
        self.data.remove_subset(s)

        self.assertNotIn(s, self.data.subsets)
        self.assertEquals(hub.broadcast.call_count, 2)

    def test_get_component(self):
        self.assertIs(self.data.get_component(self.comp_id), self.comp)

    def test_get_item(self):
        self.assertIs(self.data[self.comp_id], self.comp.data)

    def test_coordinate_links(self):
        links = self.data.coordinate_links
        w0 = self.data[self.data.get_world_component_id(0)]
        w1 = self.data[self.data.get_world_component_id(1)]
        p0 = self.data[self.data.get_pixel_component_id(0)]
        p1 = self.data[self.data.get_pixel_component_id(1)]

        w0prime = links[0].compute(self.data)
        p0prime = links[1].compute(self.data)
        w1prime = links[2].compute(self.data)
        p1prime = links[3].compute(self.data)

        np.testing.assert_array_equal(w0, w0prime)
        np.testing.assert_array_equal(w1, w1prime)
        np.testing.assert_array_equal(p0, p0prime)
        np.testing.assert_array_equal(p1, p1prime)

    def test_coordinate_links_empty_data(self):
        d = glue.core.data.Data()
        d.coords = None
        self.assertEquals(d.coordinate_links, [])

    def test_coordinate_links_idempotent(self):
        """Should only calculate links once, and
        return the same objects every time"""
        links = self.data.coordinate_links
        links2 = self.data.coordinate_links
        self.assertEquals(links, links2)

if __name__ == "__main__":
    unittest.main()