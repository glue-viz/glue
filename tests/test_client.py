import unittest

from mock import MagicMock

import glue
from glue.core.client import Client

class MockClient(Client):
    def __init__(self, *args, **kwargs):
        super(MockClient, self).__init__(*args, **kwargs)
        self.present = set()

    def _do_add_layer(self, layer):
        self.present.add(layer)

    def _do_remove_layer(self, layer):
        self.present.remove(layer)

    def _do_update_layer(self, layer):
        pass

    def layer_present(self, layer):
        return layer in self.present


class BasicClientStub(glue.core.client.BasicClient):
    def __init__(self, *args, **kwargs):
        super(BasicClientStub, self).__init__(*args, **kwargs)
        self.added = set()

    def _do_add_subset(self, subset):
        self.do_add_layer(subset)

    def _do_add_data(self, data):
        self.do_add_layer(data)

    def do_add_layer(self, layer):
        if layer in self.added:
            raise Exception("Un-caught double add")
        self.added.add(layer)

    def layer_present(self, layer):
        return layer in self.added

    def do_remove_layer(self, layer):
        if layer not in self.added:
            raise Exception("Removing non-present layer")
        self.added.remove(layer)

    def do_update_layer(self, layer):
        if layer not in self.added:
            raise Exception("Updating an absent layer")

    def _do_update_subset(self, subset):
        self.do_update_layer(subset)

    def _do_update_data(self, data):
        self.do_update_layer(data)

    def _do_remove_subset(self, subset):
        self.do_remove_layer(subset)

    def _do_remove_data(self, data):
        self.do_remove_layer(data)



class TestClient(unittest.TestCase):

    def _data(self):
        return MagicMock(spec_set = glue.core.data_collection.DataCollection)

    def _hub(self):
        return MagicMock(spec_set = glue.core.hub.Hub)

    def _client(self, data):
        return MockClient(data)

    def test_data_property(self):
        data = self._data()
        c = self._client(data)
        self.assertTrue(c.data is data)

    def test_invalid_init(self):
        self.assertRaises(TypeError, Client, None)

    def test_register(self):
        hub = self._hub()
        data = self._data()
        client = self._client(data)
        client.register_to_hub(hub)
        self.assertTrue(hub.subscribe.called)

class TestBasicClient(unittest.TestCase):

    def _create_objects(self):
        collection = glue.core.data_collection.DataCollection()
        data = glue.core.data.Data()
        subset = data.new_subset()
        collection.append(data)
        client = BasicClientStub(collection)
        return client, collection, data, subset

    def _add_subset(self):
        client, collection, data, subset = self._create_objects()
        client.add_layer(subset)
        return client, collection, data, subset

    def _add_data(self):
        client, collection, data, subset = self._create_objects()
        client.add_layer(data)
        return client, collection, data, subset

    def test_add_subset(self):
        client, collection, data, subset = self._add_subset()
        self.assertTrue(client.layer_present(subset))

    def test_data_added_with_subset(self):
        client, collection, data, subset = self._add_subset()
        self.assertTrue(client.layer_present(data))

    def test_add_data(self):
        client, collection, data, subset = self._add_data()
        self.assertTrue(client.layer_present(data))

    def test_subsets_added_with_data(self):
        client, collection, data, subset = self._add_data()
        for subset in data.subsets:
            self.assertTrue(client.layer_present(subset))

    def test_remove_subset(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(subset)
        self.assertFalse(client.layer_present(subset))

    def test_data_not_removed_with_subset(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(subset)
        self.assertTrue(client.layer_present(data))

    def test_remove_data(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(data)
        self.assertFalse(client.layer_present(data))

    def test_subsets_removed_with_data(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(data)
        self.assertFalse(client.layer_present(data))
        for subset in data.subsets:
            self.assertFalse(client.layer_present(subset))

    def test_add_subset_raises_if_not_in_collection(self):
        client, collection, data, subset = self._add_data()
        d = glue.core.data.Data()
        s = glue.core.subset.Subset(d)
        self.assertRaises(TypeError, client.add_layer, s)

    def test_add_data_raises_if_not_in_collection(self):
        client, collection, data, subset = self._add_data()
        d = glue.core.data.Data()
        s = d.new_subset()
        self.assertRaises(TypeError, client.add_layer, d)

    def test_double_add_ignored(self):
        client, collection, data, subset = self._add_subset()
        client.add_layer(subset)
        client.add_layer(data)

    def test_remove_ignored_if_not_present(self):
        client, collection, data, subset = self._add_subset()
        client.remove_layer(subset)
        client.remove_layer(subset)

    def test_update_subset_ignored_if_not_present(self):
        client, collection, data, subset = self._add_subset()
        d = glue.core.data.Data()
        s = d.new_subset()
        client.update_layer(s)

    def test_update_data_ignored_if_not_present(self):
        client, collection, data, subset = self._add_subset()
        d = glue.core.data.Data()
        s = d.new_subset()
        client.update_layer(d)

    def test_subset_messages(self):
        client, collection, data, subset = self._create_objects()
        m = MagicMock()
        m.subset = subset

        client._add_subset(m)
        self.assertTrue(client.layer_present(subset))
        client._update_subset(m)
        client._remove_subset(m)
        self.assertFalse(client.layer_present(subset))

    def test_data_messages(self):
        client, collection, data, subset = self._create_objects()
        m = MagicMock()
        m.data = data
        client.add_layer(data)
        client._update_data(m)
        client._remove_data(m)
        self.assertFalse(client.layer_present(data))

if __name__ == "__main__":
    unittest.main()