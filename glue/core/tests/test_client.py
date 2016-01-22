# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103,W0612

from __future__ import absolute_import, division, print_function

import pytest
from mock import MagicMock

from ..client import Client, BasicClient
from ..data import Data
from ..data_collection import DataCollection
from ..hub import Hub
from ..subset import Subset


class MockClient(Client):  # pylint: disable=W0223

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


class BasicClientStub(BasicClient):

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


class TestClient(object):

    def _data(self):
        return MagicMock(spec_set=DataCollection)

    def _hub(self):
        return MagicMock(spec_set=Hub)

    def _client(self, data):
        return MockClient(data)

    def test_data_property(self):
        data = self._data()
        c = self._client(data)
        assert c.data is data

    def test_invalid_init(self):
        with pytest.raises(TypeError) as exc:
            Client(None)
        assert exc.value.args[0].startswith("Input data must be a "
                                            "DataCollection:")

    def test_register(self):
        hub = self._hub()
        data = self._data()
        client = self._client(data)
        client.register_to_hub(hub)
        assert hub.subscribe.called


class TestBasicClient(object):

    def _create_objects(self):
        collection = DataCollection()
        data = Data()
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
        assert client.layer_present(subset)

    def test_data_added_with_subset(self):
        client, collection, data, subset = self._add_subset()
        assert client.layer_present(data)

    def test_add_data(self):
        client, collection, data, subset = self._add_data()
        assert client.layer_present(data)

    def test_subsets_added_with_data(self):
        client, collection, data, subset = self._add_data()
        for subset in data.subsets:
            assert client.layer_present(subset)

    def test_remove_subset(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(subset)
        assert not client.layer_present(subset)

    def test_data_not_removed_with_subset(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(subset)
        assert client.layer_present(data)

    def test_remove_data(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(data)
        assert not client.layer_present(data)

    def test_subsets_removed_with_data(self):
        client, collection, data, subset = self._add_data()
        client.remove_layer(data)
        assert not client.layer_present(data)
        for subset in data.subsets:
            assert not client.layer_present(subset)

    def test_add_subset_raises_if_not_in_collection(self):
        client, collection, data, subset = self._add_data()
        d = Data()
        s = Subset(d)
        with pytest.raises(TypeError) as exc:
            client.add_layer(s)
        assert exc.value.args[0] == "Data not in collection"

    def test_add_data_raises_if_not_in_collection(self):
        client, collection, data, subset = self._add_data()
        d = Data()
        d.new_subset()
        with pytest.raises(TypeError) as exc:
            client.add_layer(d)
        assert exc.value.args[0] == "Data not in collection"

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
        d = Data()
        s = d.new_subset()
        client.update_layer(s)

    def test_update_data_ignored_if_not_present(self):
        client, collection, data, subset = self._add_subset()
        d = Data()
        d.new_subset()
        client.update_layer(d)

    def test_subset_messages(self):
        client, collection, data, subset = self._create_objects()
        m = MagicMock()
        m.subset = subset

        client._add_subset(m)
        assert client.layer_present(subset)
        client._update_subset(m)
        client._remove_subset(m)
        assert not client.layer_present(subset)

    def test_data_messages(self):
        client, collection, data, subset = self._create_objects()
        m = MagicMock()
        m.data = data
        client.add_layer(data)
        client._update_data(m)
        client._remove_data(m)
        assert not client.layer_present(data)
