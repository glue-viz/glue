# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock
from numpy.testing import assert_array_equal, assert_equal

from ..coordinates import Coordinates
from ..component_link import ComponentLink
from ..link_helpers import LinkSame
from ..data import Data, Component, ComponentID, DerivedComponent
from ..data_collection import DataCollection
from ..hub import HubListener
from ..message import (Message, DataCollectionAddMessage, DataRemoveComponentMessage,
                       DataCollectionDeleteMessage, DataAddComponentMessage,
                       ComponentsChangedMessage, PixelAlignedDataChangedMessage)
from ..exceptions import IncompatibleAttribute

from .test_state import clone


class HubLog(HubListener):

    def __init__(self):
        self.messages = []

    def register_to_hub(self, hub):
        hub.subscribe(self, Message)

    def notify(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages[:] = []

    def messages_by_type(self, klass):
        return [msg for msg in self.messages if isinstance(msg, klass)]


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
        assert self.data not in self.dc

    def test_ignore_multi_remove(self):
        self.dc.append(self.data)
        self.dc.remove(self.data)
        self.dc.remove(self.data)
        assert self.data not in self.dc

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
        """
        When appending a data set, its DerivedComponents should be ingested into
        the LinkManager
        """
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
        """
        DerviedAttributes added to a dataset in a collection
        should generate messages that the collection catches.
        """
        d = Data()
        id1 = ComponentID("id1")
        id2 = ComponentID("id2")
        link = ComponentLink([id1], id2)
        dc = DerivedComponent(d, link)

        self.dc.append(d)
        d.add_component(Component(np.array([1, 2, 3])), id1)
        assert link not in self.dc._link_manager
        self.log.clear()
        d.add_component(dc, id2)

        assert link in self.dc._link_manager

        msgs = sorted(self.log.messages, key=lambda x: str(type(x)))

        assert isinstance(msgs[0], ComponentsChangedMessage)
        assert isinstance(msgs[1], DataAddComponentMessage)

    def test_links_auto_added(self):
        id1 = ComponentID("id1")
        id2 = ComponentID("id2")
        link = ComponentLink([id1], id2)
        self.data.links = [link]
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
        """
        Setting links attribute automatically adds components to data
        """
        d = Data()
        comp = Component(np.array([1, 2, 3]))
        id1 = ComponentID("id1")
        d.add_component(comp, id1)
        id2 = ComponentID("id2")
        self.dc.append(d)
        link = ComponentLink([id1], id2)
        self.dc.set_links([link])
        assert_equal(d[id2], d[id1])

    def test_links_propagated(self):
        """Web of links is grown and applied to data automatically"""
        from ..component_link import ComponentLink
        d = Data()
        dc = DataCollection([d])

        cid1 = d.add_component(np.array([1, 2, 3]), 'a')
        cid2 = ComponentID('b')
        cid3 = ComponentID('c')

        links1 = ComponentLink([cid1], cid2)
        dc.add_link(links1)

        assert_equal(d[cid2], d[cid1])

        links2 = ComponentLink([cid2], cid3)
        dc.add_link(links2)
        assert_equal(d[cid3], d[cid2])

        dc.remove_link(links2)
        with pytest.raises(IncompatibleAttribute):
            d[cid3]
        assert_equal(d[cid2], d[cid1])

        dc.remove_link(links1)
        with pytest.raises(IncompatibleAttribute):
            d[cid2]

    def test_merge_links(self):
        """Trivial links should be merged, discarding the duplicate ID"""
        d1 = Data(x=[1, 2, 3])
        d2 = Data(x=[2, 3, 4])
        dc = DataCollection([d1, d2])

        original_id = d2.id['x']
        link = ComponentLink([d1.id['x']], d2.id['x'])
        dc.add_link(link)

        # NOTE: the behavior tested here is not desirable anymore, so the relevant
        #       parts that are no longer true have been commented out and replaced
        #       by the new behavior.

        # assert d1.id['x'] is not original_id
        # assert duplicated_id not in d2.components
        assert d1.id['x'] is not original_id
        assert d2.id['x'] is original_id
        assert original_id in d2.components

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

    def test_merge_coordinates(self):

        # Regression test to make sure the coordinates from the first dataset
        # are preserved.

        x = Data(x=[1, 2, 3])
        y = Data(y=[2, 3, 4])
        dc = DataCollection([x, y])

        x.coords = Coordinates()
        y.coords = Coordinates()

        dc.merge(x, y)

        assert dc[0].coords is x.coords

    def test_merge_coordinates_preserve_labels(self):

        # Regression test to make sure that axis labels are preserved after
        # merging.

        x = Data(x=[1, 2, 3])
        y = Data(y=[2, 3, 4])
        dc = DataCollection([x, y])

        class CustomCoordinates(Coordinates):
            def axis_label(self, axis):
                return 'Custom {0}'.format(axis)

        x.coords = CustomCoordinates()
        y.coords = CustomCoordinates()

        dc.merge(x, y)

        assert sorted(cid.label for cid in dc[0].world_component_ids) == ['Custom 0']

    def test_merge_visible_components(self):
        # Regression test for a bug that caused visible_components to be empty
        # for a dataset made from merging other datasets.
        x = Data(x=[1, 2, 3], label='dx')
        y = Data(y=[2, 3, 4], label='dy')
        dc = DataCollection([x, y])
        dc.merge(x, y)
        assert dc[0].visible_components[0] is x.id['x']
        assert dc[0].visible_components[1] is y.id['y']

    def test_remove_component_message(self):

        # Regression test to make sure that removing a component emits the
        # appropriate messages.

        data = Data(x=[1, 2, 3], y=[4, 5, 6])
        self.dc.append(data)

        remove_id = data.id['y']

        self.log.clear()

        data.remove_component(remove_id)

        msgs = sorted(self.log.messages, key=lambda x: str(type(x)))

        print([type(msg) for msg in msgs])

        assert isinstance(msgs[0], ComponentsChangedMessage)

        assert isinstance(msgs[1], DataRemoveComponentMessage)
        assert msgs[1].component_id is remove_id

    def test_links_preserved_session(self):

        # This tests that the separation of internal vs external links is
        # preserved in session files.

        d1 = Data(a=[1, 2, 3])
        d2 = Data(b=[2, 3, 4])

        dc = DataCollection([d1, d2])
        dc.add_link(ComponentLink([d2.id['b']], d1.id['a']))

        d1['x'] = d1.id['a'] + 1

        assert len(d1.coordinate_links) == 2
        assert len(d1.derived_links) == 1
        assert len(dc._link_manager._external_links) == 1

        dc2 = clone(dc)

        assert len(dc2[0].coordinate_links) == 2
        assert len(dc2[0].derived_links) == 1
        assert len(dc2._link_manager._external_links) == 1

    def test_pixel_aligned(self):

        data1 = Data(x=np.ones((2, 4, 3)))
        data2 = Data(y=np.ones((4, 3, 2)))
        data3 = Data(z=np.ones((2, 3)))

        self.dc.extend([data1, data2, data3])

        # Add one set of links, which shouldn't change anything

        self.dc.add_link(LinkSame(data1.pixel_component_ids[0], data2.pixel_component_ids[2]))
        self.dc.add_link(LinkSame(data1.pixel_component_ids[0], data3.pixel_component_ids[0]))

        assert len(data1.pixel_aligned_data) == 0
        assert len(data2.pixel_aligned_data) == 0
        assert len(data3.pixel_aligned_data) == 0
        assert len(self.log.messages_by_type(PixelAlignedDataChangedMessage)) == 0

        # Add links between a second set of axes

        self.dc.add_link(LinkSame(data1.pixel_component_ids[2], data2.pixel_component_ids[1]))
        self.dc.add_link(LinkSame(data1.pixel_component_ids[2], data3.pixel_component_ids[1]))

        # At this point, data3 has all its axes contained in data1 and data2

        assert len(data1.pixel_aligned_data) == 0
        assert len(data2.pixel_aligned_data) == 0
        assert len(data3.pixel_aligned_data) == 2
        assert data3.pixel_aligned_data[data1] == [0, 2]
        assert data3.pixel_aligned_data[data2] == [2, 1]
        messages = self.log.messages_by_type(PixelAlignedDataChangedMessage)
        assert len(messages) == 1
        assert messages[0].data is data3

        # Finally we can link the third set of axes

        self.dc.add_link(LinkSame(data1.pixel_component_ids[1], data2.pixel_component_ids[0]))

        # At this point, data1 and data2 should now be linked

        assert len(data1.pixel_aligned_data) == 1
        assert data1.pixel_aligned_data[data2] == [2, 0, 1]
        assert len(data2.pixel_aligned_data) == 1
        assert data2.pixel_aligned_data[data1] == [1, 2, 0]
        assert len(data3.pixel_aligned_data) == 2
        assert data3.pixel_aligned_data[data1] == [0, 2]
        assert data3.pixel_aligned_data[data2] == [2, 1]
        messages = self.log.messages_by_type(PixelAlignedDataChangedMessage)
        assert len(messages) == 3
        assert messages[1].data is data1
        assert messages[2].data is data2
