import unittest

from mock import MagicMock

import glue
from glue.link_manager import LinkManager
from glue.data import ComponentID

class TestLinkManager(unittest.TestCase):
    def test_data(self):
        self.data = MagicMock()
        c1 = ComponentID('c1')
        c2 = ComponentID('c2')
        c3 = ComponentID('c3')
        c4 = ComponentID('c4')
        c5 = ComponentID('c5')
        c6 = ComponentID('c6')
        c7 = ComponentID('c7')
        c8 = ComponentID('c8')
        self.cs = [c1, c2, c3, c4, c5, c6, c6, c8]

        dummy_using = lambda x,y: (x,y)
        self.primary = [c1, c2]
        self.direct_links = [c3, c4]
        self.derived_links = [c5, c6]
        self.inaccessible_links = [c7, c8]
        self.data.primary_components = [c1, c2]
        lm = LinkManager()
        lm.make_link([c1], c3)
        lm.make_link([c2], c4)
        lm.make_link([c3], c1)
        lm.make_link([c4], c2)
        link1  = lm.make_link([c3, c4], c5, dummy_using)
        link2 = lm.make_link([c3, c4], c6, dummy_using)
        self.expected_derived_links = [link1, link2]
        self.short_path_test_id = c5
        self.expected_short_path_link = link1
        self.lm = lm

    def short_circuit(self):
        link = self.lm.make_link([self.cs[0]], self.cs[4])
        self.expected_short_path_link = link

    def test_make_links(self):
        id1 = ComponentID('id1')
        id2 = ComponentID('id2')
        id3 = ComponentID('id3')
        lm = LinkManager()
        using = lambda x,y: 0
        link = lm.make_link([id1, id2], id3, using)
        links = lm.links
        self.assertEquals(links, [link])
        self.assertEquals(link.get_from_ids(), [id1, id2])
        self.assertEquals(link.get_to_id(), id3)
        self.assertEquals(link.get_using(), using)

    def test_virtual_components(self):
        self.test_data()
        virtual = self.lm.virtual_components(self.data)

        for inaccessible in self.inaccessible_links:
            self.assertFalse(inaccessible in virtual)

        for direct in self.direct_links:
            self.assertTrue(direct in virtual)

        for direct in self.derived_links:
            self.assertTrue(direct in virtual)

        for direct in self.primary:
            self.assertFalse(direct in virtual)

    def test_virtual_components_point_to_proper_ids(self):
        self.test_data()
        virtual = self.lm.virtual_components(self.data)

        for v in virtual:
            self.assertEquals(v, virtual[v].get_to_id())

    def test_virtual_compoents_choose_shortest_path(self):
        self.test_data()
        virtual = self.lm.virtual_components(self.data)
        link1 = virtual[self.short_path_test_id]
        self.assertIs(link1, self.expected_short_path_link)

        self.short_circuit()
        virtual = self.lm.virtual_components(self.data)
        link2 = virtual[self.short_path_test_id]

        self.assertTrue(link1 is not link2)
        self.assertIs(link2, self.expected_short_path_link)


    def test_virtual_compoents_choose_right_links(self):
        data = self.test_data()
        virtual = self.lm.virtual_components(self.data)

        for l, cid in zip(self.expected_derived_links, self.derived_links):
            self.assertIs(virtual[cid], l)



if __name__ == "__main__":
    unittest.main()