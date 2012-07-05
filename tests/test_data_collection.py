import unittest

import numpy as np
from mock import MagicMock

import glue

class HubLog(glue.core.hub.HubListener):
    def __init__(self):
        self.messages = []

    def register_to_hub(self, hub):
        hub.subscribe(self, glue.core.message.Message)

    def notify(self, message):
        self.messages.append(message)

class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.dc = glue.core.data_collection.DataCollection()
        self.data = MagicMock()
        self.hub = glue.core.hub.Hub()
        self.log = HubLog()
        self.log.register_to_hub(self.hub)

    def test_init(self):
        d = glue.core.data.Data()
        dc = glue.core.data_collection.DataCollection(d)
        assert d in dc
        dc = glue.core.data_collection.DataCollection([d])
        assert d in dc

    def test_data(self):
        self.dc.append(self.data)
        assert self.dc.data == [self.data]

    def test_append(self):
        self.dc.append(self.data)
        assert self.data in self.dc

    def test_ignore_multi_add(self):
        self.dc.append(self.data)
        self.dc.append(self.data)
        self.assertEquals(len(self.dc), 1)

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
        self.dc.register_to_hub(self.hub)
        self.dc.append(self.data)
        msg = self.log.messages[-1]
        assert msg.sender == self.dc
        self.assertIsInstance(msg is glue.core.message.DataCollectionAddMessage
        assert msg.data is self.data

    def test_remove_broadcast(self):
        self.dc.register_to_hub(self.hub)
        self.dc.append(self.data)
        self.dc.remove(self.data)
        msg = self.log.messages[-1]
        assert msg.sender == self.dc
        self.assertIsInstance(msg is glue.core.message.DataCollectionDeleteMessage
        assert msg.data is self.data

    def test_register_adds_hub(self):
        self.dc.register_to_hub(self.hub)
        assert self.dc.hub is self.hub

    def test_register_assigns_hub_of_data(self):
        self.dc.append(self.data)
        self.dc.register_to_hub(self.hub)
        self.data.register_to_hub.assert_called_once_with(self.hub)

    def test_get_item(self):
        self.dc.append(self.data)
        assert self.dc[0] is self.data

    def test_iter(self):
        self.dc.append(self.data)
        self.assertEquals(set(self.dc), set([self.data]))

    def test_len(self):
        self.assertEquals(len(self.dc), 0)
        self.dc.append(self.data)
        self.assertEquals(len(self.dc), 1)
        self.dc.append(self.data)
        self.assertEquals(len(self.dc), 1)
        self.dc.remove(self.data)
        self.assertEquals(len(self.dc), 0)

    def test_derived_links_autoadd(self):
        """When appending a data set, its DerivedComponents
        should be ingested into the LinkManager"""
        d = glue.core.data.Data()
        id1 = glue.core.data.ComponentID("id1")
        id2 = glue.core.data.ComponentID("id2")
        link = glue.core.component_link.ComponentLink([id1], id2)
        dc = glue.core.data.DerivedComponent(d, link)
        d.add_component(glue.core.data.Component(np.array([1,2,3])), id1)
        d.add_component(dc, id2)

        dc = glue.core.data_collection.DataCollection()
        dc.append(d)

        assert link in dc._link_manager

    def test_catch_data_add_component_message(self):
        """DerviedAttributes added to a dataset in a collection
        should generate messages that the collection catches.
        """
        d = glue.core.data.Data()
        id1 = glue.core.data.ComponentID("id1")
        id2 = glue.core.data.ComponentID("id2")
        link = glue.core.component_link.ComponentLink([id1], id2)
        dc = glue.core.data.DerivedComponent(d, link)

        self.dc.register_to_hub(self.hub)
        self.dc.append(d)
        d.add_component(glue.core.data.Component(np.array([1,2,3])), id1)
        assert not link in self.dc._link_manager
        d.add_component(dc, id2)

        msg = self.log.messages[-1]
        self.assertIsInstance(msg is glue.core.message.DataAddComponentMessage
        assert link in self.dc._link_manager

    def test_coordinate_links_auto_added(self):
        d = glue.core.data.Data()
        id1 = glue.core.data.ComponentID("id1")
        id2 = glue.core.data.ComponentID("id2")
        link = glue.core.component_link.ComponentLink([id1], id2)
        self.data.coordinate_links = [link]
        self.dc.append(self.data)
        assert link in self.dc._link_manager.links


if __name__ == "__main__":
    unittest.main()