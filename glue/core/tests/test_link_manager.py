import numpy as np

from ..data import Data, Component
from ..component_link import ComponentLink
from ..link_manager import LinkManager, accessible_links, discover_links, \
                           find_dependents
from ..data import ComponentID, DerivedComponent

comp = Component(data=np.array([1, 2, 3]))


def example_components(self, add_derived=True):
    """ Link Topology

           --- c1---c3--\
    data --|             --c5,c6   (c7,c8 disconnected)
           --- c2---c4--/
    """
    self.data = Data()
    c1 = ComponentID('c1')
    c2 = ComponentID('c2')
    c3 = ComponentID('c3')
    c4 = ComponentID('c4')
    c5 = ComponentID('c5')
    c6 = ComponentID('c6')
    c7 = ComponentID('c7')
    c8 = ComponentID('c8')

    dummy_using = lambda x, y: (x, y)
    self.cs = [c1, c2, c3, c4, c5, c6, c7, c8]
    self.links = [ComponentLink([c1], c3),
                  ComponentLink([c2], c4),
                  ComponentLink([c3], c1),
                  ComponentLink([c4], c2),
                  ComponentLink([c3, c4], c5, dummy_using),
                  ComponentLink([c3, c4], c6, dummy_using)]

    self.data.add_component(comp, c1)
    self.data.add_component(comp, c2)
    if add_derived:
        for i in [0, 1, 4, 5]:
            dc = DerivedComponent(self.data, self.links[i])
            self.data.add_component(dc, dc.link.get_to_id())

    self.primary = [c1, c2]
    self.direct = [c3, c4]
    self.derived = [c5, c6]
    self.inaccessible = [c7, c8]


class TestAccessibleLinks(object):

    def setup_method(self, method):
        self.cs = [ComponentID("%i" % i) for i in xrange(10)]

    def test_returned_if_available(self):
        cids = self.cs[0:5]
        links = [ComponentLink([self.cs[0]], self.cs[1])]
        assert links[0] in accessible_links(cids, links)

    def test_returned_if_reachable(self):
        cids = self.cs[0:5]
        links = [ComponentLink([self.cs[0]], self.cs[6])]
        assert links[0] in accessible_links(cids, links)

    def test_not_returned_if_not_reachable(self):
        cids = self.cs[0:5]
        links = [ComponentLink([self.cs[6]], self.cs[7])]
        assert not links[0] in accessible_links(cids, links)


class TestDiscoverLinks(object):
    def setup_method(self, method):
        example_components(self)

    def test_correct_discover(self):
        """discover_links finds the correct links"""
        links = discover_links(self.data, self.links)

        for i in self.inaccessible:
            assert not i in links

        for d in self.direct:
            assert d in links

        for d in self.derived:
            assert d in links

        for p in self.primary:
            assert not p in links

    def test_links_point_to_proper_ids(self):
        """ Dictionary values are ComponentLinks which
        point to the keys """
        links = discover_links(self.data, self.links)
        for cid in links:
            assert cid == links[cid].get_to_id()

    def test_shortest_path(self):
        """ Shortcircuit c5 to c1, yielding 2 ways to get to c5.
        Ensure that the shortest path is chosen """
        self.links.append(ComponentLink([self.cs[0]], self.cs[4]))
        links = discover_links(self.data, self.links)

        assert links[self.cs[4]] is self.links[-1]


class TestFindDependents(object):
    def setup_method(self, method):
        example_components(self)

    def test_propagated(self):
        to_remove = self.links[0]
        result = find_dependents(self.data, to_remove)
        expected = set([self.cs[2], self.cs[4], self.cs[5]])
        assert expected == result

    def test_basic(self):
        to_remove = self.links[4]
        result = find_dependents(self.data, to_remove)
        expected = set([self.cs[4]])
        assert expected == result


class TestLinkManager(object):

    def test_add_links(self):
        id1 = ComponentID('id1')
        id2 = ComponentID('id2')
        id3 = ComponentID('id3')
        lm = LinkManager()
        using = lambda x, y: 0
        link = ComponentLink([id1, id2], id3, using)
        lm.add_link(link)
        links = lm.links
        assert links == [link]

    def test_remove_link(self):
        id1 = ComponentID('id1')
        id2 = ComponentID('id2')
        id3 = ComponentID('id3')
        lm = LinkManager()
        using = lambda x, y: 0
        link = ComponentLink([id1, id2], id3, using)
        lm.add_link(link)
        lm.remove_link(link)
        links = lm.links
        assert links == []

    def test_setup(self):
        example_components(self, add_derived=False)
        expected = set()
        assert set(self.data.derived_components) == expected

    def test_update_data_components_adds_correctly(self):
        example_components(self, add_derived=False)
        lm = LinkManager()
        map(lm.add_link, self.links)

        lm.update_data_components(self.data)
        derived = set(self.data.derived_components)
        expected = set(self.direct + self.derived)
        assert derived == expected

    def test_update_data_components_removes_correctly(self):
        #add all but last link to manager
        example_components(self, add_derived=False)
        lm = LinkManager()
        map(lm.add_link, self.links[:-1])

        #manually add last link as derived component
        dc = DerivedComponent(self.data, self.links[-1])
        self.data.add_component(dc, dc.link.get_to_id())
        removed = set([dc.link.get_to_id()])
        assert dc.link.get_to_id() in self.data.derived_components

        # this link should be removed upon update_components
        lm.update_data_components(self.data)
        derived = set(self.data.derived_components)
        expected = set(self.direct + self.derived) - removed
        assert derived == expected
