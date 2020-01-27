# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from itertools import product

import numpy as np
from numpy.testing import assert_array_equal

from ..component_link import ComponentLink
from ..data import ComponentID, DerivedComponent, Data, Component
from ..coordinates import IdentityCoordinates
from ..data_collection import DataCollection
from ..link_manager import (LinkManager, accessible_links, discover_links,
                            find_dependents, is_convertible_to_single_pixel_cid,
                            equivalent_pixel_cids, pixel_cid_to_pixel_cid_matrix)
from ..link_helpers import LinkSame, MultiLink

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
    self.links = [ComponentLink([c1], c3, lambda x:x),
                  ComponentLink([c2], c4, lambda x:x),
                  ComponentLink([c3], c1, lambda x:x),
                  ComponentLink([c4], c2, lambda x:x),
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
    self.derived = []
    self.externally_derived = [c5, c6]
    self.inaccessible = [c7, c8]


class TestAccessibleLinks(object):

    def setup_method(self, method):
        self.cs = [ComponentID("%i" % i) for i in range(10)]

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
            assert i not in links

        for d in self.direct:
            assert d in links

        for d in self.derived:
            assert d in links

        for p in self.primary:
            assert p not in links

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

    def test_update_externally_derivable_components_adds_correctly(self):
        example_components(self, add_derived=False)
        lm = LinkManager()
        list(map(lm.add_link, self.links))

        lm.update_externally_derivable_components(self.data)
        derived = set(self.data.externally_derivable_components)
        expected = set(self.externally_derived + self.direct)
        assert derived == expected

    def test_update_externally_derivable_components_removes_correctly(self):
        # add all but last link to manager
        example_components(self, add_derived=False)
        lm = LinkManager()
        list(map(lm.add_link, self.links[:-1]))
        lm.update_externally_derivable_components(self.data)

        # manually add last link as derived component
        dc = DerivedComponent(self.data, self.links[-1])
        self.data._externally_derivable_components.update({dc.link.get_to_id(): dc})
        removed = set([dc.link.get_to_id()])
        assert dc.link.get_to_id() in self.data.externally_derivable_components

        # this link should be removed upon update_components
        lm.update_externally_derivable_components(self.data)
        derived = set(self.data.externally_derivable_components)
        expected = set(self.direct + self.externally_derived) - removed
        assert derived == expected

    def test_derived_links_correctwith_mergers(self):
        """When the link manager merges components, links that depend on the
        merged components remain functional"""

        d1 = Data(x=[[1, 2], [3, 4]], coords=IdentityCoordinates(n_dim=2))
        d2 = Data(u=[[5, 6], [7, 8]], coords=IdentityCoordinates(n_dim=2))

        dc = DataCollection([d1, d2])

        # link world coordinates...
        dc.add_link(LinkSame(
            d1.world_component_ids[0], d2.world_component_ids[0]))
        dc.add_link(LinkSame(
            d1.world_component_ids[1], d2.world_component_ids[1]))

        # and then retrieve pixel coordinates
        assert_array_equal(d2[d1.pixel_component_ids[0]], [[0, 0], [1, 1]])
        assert_array_equal(d1[d2.pixel_component_ids[1]], [[0, 1], [0, 1]])

    def test_binary_links_correct_with_mergers(self):
        """Regression test. BinaryComponentLinks should work after mergers"""

        d1 = Data(x=[1, 2, 3], y=[2, 3, 4])
        d2 = Data(u=[2, 3, 4], v=[3, 4, 5])

        z = d1.id['x'] + d1.id['y']
        d1.add_component_link(z, 'z')

        dc = DataCollection([d1, d2])
        dc.add_link(LinkSame(d2.id['u'], d1.id['x']))

        assert_array_equal(d1['z'], [3, 5, 7])

    def test_complex_links_correct_with_mergers(self):
        """Regression test. multi-level links should work after mergers"""

        d1 = Data(x=[1, 2, 3], y=[2, 3, 4])
        d2 = Data(u=[2, 3, 4], v=[3, 4, 5])
        x = d1.id['x']

        z = d1.id['x'] + d1.id['y'] + 5
        d1.add_component_link(z, 'z')

        dc = DataCollection([d1, d2])
        dc.add_link(LinkSame(d2.id['u'], d1.id['x']))

        # NOTE: the behavior tested here is not desirable anymore, so the
        #       following assert is no longer True
        # assert x not in d1.components

        assert_array_equal(d1['z'], [8, 10, 12])

    def test_remove_data_removes_links(self):

        d1 = Data(x=[[1, 2], [3, 4]], label='image', coords=IdentityCoordinates(n_dim=2))
        d2 = Data(a=[1, 2, 3], b=[4, 5, 6], label='catalog', coords=IdentityCoordinates(n_dim=3))

        dc = DataCollection([d1, d2])

        assert len(dc.links) == 6

        dc.add_link(LinkSame(d1.id['x'], d2.id['a']))

        assert len(dc.links) == 7

        # Removing dataset should remove related links
        dc.remove(d1)
        assert len(dc.links) == 2

    def test_remove_component_removes_links(self):

        d1 = Data(x=[[1, 2], [3, 4]], label='image', coords=IdentityCoordinates(n_dim=2))
        d2 = Data(a=[1, 2, 3], b=[4, 5, 6], label='catalog', coords=IdentityCoordinates(n_dim=3))

        dc = DataCollection([d1, d2])

        assert len(dc.links) == 6

        assert len(d1.components) == 5
        assert len(d2.components) == 4

        assert len(d1.externally_derivable_components) == 0
        assert len(d2.externally_derivable_components) == 0

        dc.add_link(LinkSame(d1.id['x'], d2.id['a']))

        assert len(dc.links) == 7

        assert len(d1.components) == 5
        assert len(d2.components) == 4

        assert len(d1.externally_derivable_components) == 1
        assert len(d2.externally_derivable_components) == 1

        assert d1.id['x'] in d2.externally_derivable_components
        assert d2.id['a'] in d1.externally_derivable_components

        # Removing component a from d2 should remove related links and
        # derived components.
        d2.remove_component(d2.id['a'])

        assert len(dc.links) == 6

        assert len(d1.components) == 5
        assert len(d2.components) == 3

        assert len(d1.externally_derivable_components) == 0
        assert len(d2.externally_derivable_components) == 0


def test_is_convertible_to_single_pixel_cid():

    # This tests the function is_convertible_to_single_pixel_cid, which gives
    # for a given dataset the pixel component ID that can be uniquely
    # transformed into the requested component ID.

    # Set up a coordinate object which has an independent first axis and
    # has the second and third axes depend on each other. The transformation
    # itself is irrelevant since for this function to work we only care about
    # whether or not an axis is independent.

    class CustomCoordinates(IdentityCoordinates):

        @property
        def axis_correlation_matrix(self):
            matrix = np.zeros((self.world_n_dim, self.pixel_n_dim), dtype=bool)
            matrix[2, 2] = True
            matrix[0:2, 0:2] = True
            return matrix

    data1 = Data()
    data1.coords = CustomCoordinates(n_dim=3)
    data1['x'] = np.ones((4, 3, 4))
    px1, py1, pz1 = data1.pixel_component_ids
    wx1, wy1, wz1 = data1.world_component_ids
    data1['a'] = px1 * 2
    data1['b'] = wx1 * 2
    data1['c'] = wy1 * 2
    data1['d'] = wx1 * 2 + px1
    data1['e'] = wx1 * 2 + wy1

    # Pixel component IDs should just be returned directly
    for cid in data1.pixel_component_ids:
        assert is_convertible_to_single_pixel_cid(data1, cid) is cid

    # Only the first world component should return a valid pixel component
    # ID since the two other world components are interlinked
    assert is_convertible_to_single_pixel_cid(data1, wx1) is px1
    assert is_convertible_to_single_pixel_cid(data1, wy1) is None
    assert is_convertible_to_single_pixel_cid(data1, wz1) is None

    # a and b are ultimately linked to the first pixel coordinate, whereas c
    # depends on the second world coordinate which is interlinked with the third
    # Finally, d is ok because it really only depends on px1
    assert is_convertible_to_single_pixel_cid(data1, data1.id['a']) is px1
    assert is_convertible_to_single_pixel_cid(data1, data1.id['b']) is px1
    assert is_convertible_to_single_pixel_cid(data1, data1.id['c']) is None
    assert is_convertible_to_single_pixel_cid(data1, data1.id['d']) is px1
    assert is_convertible_to_single_pixel_cid(data1, data1.id['e']) is None

    # We now create a second dataset and set up links
    data2 = Data(y=np.ones((4, 5, 6, 7)), z=np.zeros((4, 5, 6, 7)))
    dc = DataCollection([data1, data2])
    dc.add_link(ComponentLink([data1.id['a'], px1], data2.id['y'], using=lambda x: 2 * x))
    dc.add_link(ComponentLink([wy1], data2.id['z'], using=lambda x: 2 * x))

    assert is_convertible_to_single_pixel_cid(data1, data2.id['y']) is px1
    assert is_convertible_to_single_pixel_cid(data1, data2.id['z']) is None


def test_equivalent_pixel_cids():

    # This tests the equivalent_pixel_cids function which checks whether all
    # pixel component IDs in a dataset are equivalent to pixel component IDs
    # in another dataset.

    data1 = Data(x=np.ones((2, 4, 3)))
    data2 = Data(y=np.ones((4, 3, 2)))
    data3 = Data(z=np.ones((2, 3)))

    dc = DataCollection([data1, data2, data3])

    # Checking data against itself should give the normal axis order

    assert equivalent_pixel_cids(data1, data1) == [0, 1, 2]
    assert equivalent_pixel_cids(data3, data3) == [0, 1]

    # Checking data against unlinked data should give None

    for d1, d2 in product(dc, dc):
        if d1 is not d2:
            assert equivalent_pixel_cids(d1, d2) is None

    # Add one set of links, which shouldn't change anything

    dc.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[2]))
    dc.add_link(LinkSame(data1.pixel_component_ids[0], data3.pixel_component_ids[0]))

    for d1, d2 in product(dc, dc):
        if d1 is not d2:
            assert equivalent_pixel_cids(d1, d2) is None

    # Add links between a second set of axes

    dc.add_link(LinkSame(data1.pixel_component_ids[2], data2.pixel_component_ids[1]))
    dc.add_link(LinkSame(data1.pixel_component_ids[2], data3.pixel_component_ids[1]))

    # At this point, data3 has all its axes contained in data1 and data2

    assert equivalent_pixel_cids(data1, data2) is None
    assert equivalent_pixel_cids(data2, data1) is None
    assert equivalent_pixel_cids(data1, data3) == [0, 2]
    assert equivalent_pixel_cids(data3, data1) is None
    assert equivalent_pixel_cids(data2, data3) == [2, 1]
    assert equivalent_pixel_cids(data3, data2) is None

    # Finally we can link the third set of axes

    dc.add_link(LinkSame(data1.pixel_component_ids[1], data2.pixel_component_ids[0]))

    # At this point, data1 and data2 should now be linked. Note that in cases
    # where the target has more dimensions than the reference, we always get None

    assert equivalent_pixel_cids(data1, data2) == [1, 2, 0]
    assert equivalent_pixel_cids(data2, data1) == [2, 0, 1]
    assert equivalent_pixel_cids(data1, data3) == [0, 2]
    assert equivalent_pixel_cids(data3, data1) is None
    assert equivalent_pixel_cids(data2, data3) == [2, 1]
    assert equivalent_pixel_cids(data3, data2) is None


def test_pixel_cid_to_pixel_cid_matrix():

    # This tests the pixel_cid_to_pixel_cid_matrix function which returns
    # a correlation matrix between the pixel components from two dataset.

    data1 = Data(x=np.ones((2, 4, 3)))
    data2 = Data(y=np.ones((4, 3, 2)))
    data3 = Data(z=np.ones((2, 3)))

    dc = DataCollection([data1, data2, data3])

    # Checking data against itself should give an identity matrix

    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data1, data1), np.identity(3))
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data2, data2), np.identity(3))
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data3, data3), np.identity(2))

    # Checking data against unlinked data should give None

    for d1, d2 in product(dc, dc):
        if d1 is not d2:
            assert_array_equal(pixel_cid_to_pixel_cid_matrix(d1, d2),
                               np.zeros((d1.ndim, d2.ndim)))

    # Add one set of links

    dc.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[2]))
    dc.add_link(LinkSame(data1.pixel_component_ids[0], data3.pixel_component_ids[0]))

    expected = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data1, data2), expected)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data2, data1), expected.T)

    expected = np.array([[1, 0], [0, 0], [0, 0]], dtype=bool)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data1, data3), expected)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data3, data1), expected.T)

    expected = np.array([[0, 0], [0, 0], [1, 0]], dtype=bool)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data2, data3), expected)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data3, data2), expected.T)

    # Add links with multiple components, in this case linking two cids with two cids

    def forwards(x, y):
        return x + y, x - y

    def backwards(x, y):
        return 0.5 * (x + y), 0.5 * (x - y)

    dc.add_link(MultiLink(data1.pixel_component_ids[1:],
                          data2.pixel_component_ids[:2],
                          forwards=forwards, backwards=backwards))

    expected = np.array([[0, 0, 1], [1, 1, 0], [1, 1, 0]], dtype=bool)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data1, data2), expected)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data2, data1), expected.T)

    expected = np.array([[1, 0], [0, 0], [0, 0]], dtype=bool)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data1, data3), expected)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data3, data1), expected.T)

    expected = np.array([[0, 0], [0, 0], [1, 0]], dtype=bool)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data2, data3), expected)
    assert_array_equal(pixel_cid_to_pixel_cid_matrix(data3, data2), expected.T)
