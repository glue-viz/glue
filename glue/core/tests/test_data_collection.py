# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock
from numpy.testing import assert_array_equal

from ..component_link import ComponentLink
from ..data import Data, Component, ComponentID, DerivedComponent
from ..data_collection import DataCollection
from ..hub import HubListener
from ..message import (Message, DataCollectionAddMessage,
                       DataCollectionDeleteMessage,
                       ComponentsChangedMessage)


class HubLog(HubListener):

    def __init__(self):
        self.messages = []

    def register_to_hub(self, hub):
        hub.subscribe(self, Message)

    def notify(self, message):
        self.messages.append(message)


class TestDataCollection(object):

    def setup_method(self, method):
        self.dc = DataCollection()
        self.data = MagicMock(spec_set=Data)
        self.hub = self.dc.hub
        self.log = HubLog()
        self.log.register_to_hub(self.hub)

    def test_init_scalar(self):
        """Single data object passed to init adds to collection"""
        d = Data()
        dc = DataCollection(d)
        assert d in dc

    def test_init_list(self):
        """List of data objects passed to init auto-added to collection"""
        d1 = Data()
        d2 = Data()
        dc = DataCollection([d1, d2])
        assert d1 in dc
        assert d2 in dc

    def test_data(self):
        """ data attribute is a list of all appended data"""
        self.dc.append(self.data)
        assert self.dc.data == [self.data]

    def test_append(self):
        """ append method adds to collection """
        self.dc.append(self.data)
        assert self.data in self.dc

    def test_multi_append(self):
        """ append method works with lists """
        d = Data('test1', x=[1, 2, 3])
        d2 = Data('test2', y=[2, 3, 4])
        self.dc.append([d, d2])
        assert d in self.dc
        assert d2 in self.dc

    def test_ignore_multi_add(self):
        """ data only added once, even after multiple calls to append """
        self.dc.append(self.data)
        self.dc.append(self.data)
        assert len(self.dc) == 1

    def test_remove(self):
        self.dc.append(self.data)
        self.dc.remove(self.data)
        assert not self.data in self.dc

    def test_ignore_multi_remove(self):
        self.dc.append(self.data)
        self.dc.remove(self.data)
        self.dc.remove(self.data)
        assert not self.data in self.dc

    def test_append_broadcast(self):
        """ Call to append generates a DataCollectionAddMessage """
        self.dc.append(self.data)
        msg = self.log.messages[-1]
        assert msg.sender == self.dc
        assert isinstance(msg, DataCollectionAddMessage)
        assert msg.data is self.data

    def test_remove_broadcast(self):
        """ call to remove generates a DataCollectionDeleteMessage """
        self.dc.append(self.data)
        self.dc.remove(self.data)
        msg = self.log.messages[-1]
        assert msg.sender == self.dc
        assert isinstance(msg, DataCollectionDeleteMessage)
        assert msg.data is self.data

    def test_register_assigns_hub_of_data(self):
        self.dc.append(self.data)
        self.data.register_to_hub.assert_called_once_with(self.hub)

    def test_get_item(self):
        self.dc.append(self.data)
        assert self.dc[0] is self.data

    def test_iter(self):
        self.dc.append(self.data)
        assert set(self.dc) == set([self.data])

    def test_len(self):
        assert len(self.dc) == 0
        self.dc.append(self.data)
        assert len(self.dc) == 1
        self.dc.append(self.data)
        assert len(self.dc) == 1
        self.dc.remove(self.data)
        assert len(self.dc) == 0

    def test_derived_links_autoadd(self):
        """When appending a data set, its DerivedComponents
        should be ingested into the LinkManager"""
        d = Data()
        id1 = ComponentID("id1")
        id2 = ComponentID("id2")
        link = ComponentLink([id1], id2)
        dc = DerivedComponent(d, link)
        d.add_component(Component(np.array([1, 2, 3])), id1)
        d.add_component(dc, id2)

        dc = DataCollection()
        dc.append(d)

        assert link in dc._link_manager

    def test_catch_data_add_component_message(self):
        """DerviedAttributes added to a dataset in a collection
        should generate messages that the collection catches.
        """
        d = Data()
        id1 = ComponentID("id1")
        id2 = ComponentID("id2")
        link = ComponentLink([id1], id2)
        dc = DerivedComponent(d, link)

        self.dc.append(d)
        d.add_component(Component(np.array([1, 2, 3])), id1)
        assert not link in self.dc._link_manager
        d.add_component(dc, id2)

        msg = self.log.messages[-1]
        assert isinstance(msg, ComponentsChangedMessage)
        assert link in self.dc._link_manager

    def test_coordinate_links_auto_added(self):
        id1 = ComponentID("id1")
        id2 = ComponentID("id2")
        link = ComponentLink([id1], id2)
        self.data.coordinate_links = [link]
        self.dc.append(self.data)
        assert link in self.dc.links

    def test_add_links(self):
        """ links attribute behaves like an editable list """
        id1 = ComponentID("id1")
        id2 = ComponentID("id2")
        link = ComponentLink([id1], id2)
        self.dc.set_links([link])
        assert link in self.dc.links

    def test_add_links_updates_components(self):
        """setting links attribute automatically adds components to data"""
        d = Data()
        comp = MagicMock(spec_set=Component)
        id1 = ComponentID("id1")
        d.add_component(comp, id1)
        id2 = ComponentID("id2")
        self.dc.append(d)
        link = ComponentLink([id1], id2, using=lambda x: None)

        self.dc.set_links([link])
        assert id2 in d.components

    def test_links_propagated(self):
        """Web of links is grown and applied to data automatically"""
        from ..component_link import ComponentLink
        d = Data()
        dc = DataCollection([d])

        cid1 = d.add_component(np.array([1, 2, 3]), 'a')
        cid2 = ComponentID('b')
        cid3 = ComponentID('c')

        dummy = lambda x: None
        links = ComponentLink([cid1], cid2, dummy)
        dc.add_link(links)
        assert cid2 in d.components

        links = ComponentLink([cid2], cid3, dummy)
        dc.add_link(links)
        assert cid3 in d.components

    def test_merge_links(self):
        """Trivial links should be merged, discarding the duplicate ID"""
        d1 = Data(x=[1, 2, 3])
        d2 = Data(x=[2, 3, 4])
        dc = DataCollection([d1, d2])

        duplicated_id = d2.id['x']
        link = ComponentLink([d1.id['x']], d2.id['x'])
        dc.add_link(link)

        assert d1.id['x'] is d2.id['x']
        assert d1.id['x'] is not duplicated_id
        assert duplicated_id not in d2.components

        assert_array_equal(d1[d1.id['x']], [1, 2, 3])
        assert_array_equal(d2[d1.id['x']], [2, 3, 4])

    def test_merge(self):

        x = Data(x=[1, 2, 3])
        y = Data(y=[2, 3, 4])
        dc = DataCollection([x, y])

        dc.merge(x, y)

        assert x not in dc
        assert y not in dc

        assert_array_equal(dc[0]['x'], [1, 2, 3])
        assert_array_equal(dc[0]['y'], [2, 3, 4])

    def test_merge_discards_duplicate_pixel_components(self):
        x = Data(x=[1, 2, 3])
        y = Data(y=[2, 3, 4])
        dc = DataCollection([x, y])
        dc.merge(x, y)

        assert y.pixel_component_ids[0] not in x.components

    def test_merge_forbids_single_argument(self):
        x = Data(x=[1, 2, 3])
        y = Data(y=[2, 3, 4])
        dc = DataCollection([x, y])
        with pytest.raises(ValueError) as exc:
            dc.merge(x)
        assert exc.value.args[0] == 'merge requires 2 or more arguments'

    def test_merge_requires_same_shapes(self):
        x = Data(x=[1, 2, 3])
        y = Data(y=[2, 3, 4, 5])
        dc = DataCollection([x, y])
        with pytest.raises(ValueError) as exc:
            dc.merge(x, y)
        assert exc.value.args[0] == 'All arguments must have the same shape'

    def test_merge_disambiguates_components(self):
        x = Data(x=[1, 2, 3])
        old = set(x.components)
        y = Data(x=[2, 3, 4])
        dc = DataCollection([x, y])
        dc.merge(x, y)

        z = dc[0]
        new = list(set(z.components) - old)[0]

        assert new.label != 'x'

    def test_merge_multiargument(self):

        dc = DataCollection([Data(x=[1, 2, 3]),
                             Data(y=[2, 3, 4]),
                             Data(z=[3, 4, 5])])

        dc.merge(*list(dc))
        assert len(dc) == 1
        d = dc[0]
        assert_array_equal(d['y'], [2, 3, 4])
        assert_array_equal(d['z'], [3, 4, 5])

    def test_merging_preserves_links_forwards(self):

        a = Data(a=[1, 2, 3])
        b = Data(b=[2, 3, 4])
        c = Data(c=[3, 4, 5])

        dc = DataCollection([a, b, c])
        dc.add_link(ComponentLink([a.id['a']], b.id['b'], lambda x: x))
        dc.add_link(ComponentLink([b.id['b']], c.id['c'], lambda x: x))

        assert_array_equal(a['c'], [1, 2, 3])
        dc.merge(a, b)
        assert_array_equal(a['c'], [1, 2, 3])

    def test_merging_preserves_links_backwards(self):

        a = Data(a=[1, 2, 3])
        b = Data(b=[2, 3, 4])
        c = Data(c=[3, 4, 5])

        dc = DataCollection([a, b, c])
        dc.add_link(ComponentLink([c.id['c']], b.id['b'], lambda x: x))
        dc.add_link(ComponentLink([b.id['b']], a.id['a'], lambda x: x))

        assert_array_equal(c['a'], [3, 4, 5])
        dc.merge(a, b)
        assert_array_equal(c['a'], [3, 4, 5])
