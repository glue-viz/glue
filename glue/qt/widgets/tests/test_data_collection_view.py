#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from ..data_collection_view import DataCollectionView
from .... import core
from ....tests import example_data

from PyQt4.QtGui import QApplication


class Printer(core.hub.HubListener):
    def register_to_hub(self, hub):
        hub.subscribe(self, core.message.Message, self.print_msg)

    def print_msg(self, message):
        print message


class TestDataCollectionView(object):

    def setup_method(self, method):
        self.data = example_data.test_data()[0]
        self.collect = core.DataCollection()
        self.hub = core.hub.Hub()
        self.view = DataCollectionView()
        self.view.setup(self.collect, self.hub)
        p = Printer()
        p.register_to_hub(self.hub)

    def has_item(self, layer):
        return layer in self.view

    def get_item(self, layer):
        return self.view[layer]

    def test_add_data_updates_view(self):
        self.collect.append(self.data)
        self.view._assert_view_synced()
        assert self.has_item(self.data)

    def test_add_data_updates_subset_view(self):
        """subsets should be auto-added with data"""
        self.collect.append(self.data)
        for subset in self.data.subsets:
            assert self.has_item(subset)

    def test_add_subset_updates_view(self):
        self.collect.append(self.data)
        subset = self.data.new_subset()
        assert self.has_item(subset)

    def test_data_autoadded_on_setup(self):
        self.collect.append(self.data)
        widget = DataCollectionView()
        widget.setup(self.collect, self.hub)
        assert self.data in widget
        for s in self.data.subsets:
            assert s in widget

    def test_data_delete_updates_view(self):
        self.collect.append(self.data)
        self.collect.remove(self.data)
        assert not self.has_item(self.data)

    def test_subset_delete_updates_view(self):
        self.collect.append(self.data)
        s = self.data.new_subset()
        assert self.has_item(s)
        s.delete()
        assert not self.has_item(s)

    def test_data_delete_removes_subsets(self):
        self.collect.append(self.data)
        s = self.data.new_subset()
        self.collect.remove(self.data)
        assert not self.has_item(s)

    def test_update_data_label(self):
        self.collect.append(self.data)
        self.data.label = "testing"
        item = self.get_item(self.data)
        assert item.text(0) == "testing"

    def test_update_subset_label(self):
        self.collect.append(self.data)
        subset = self.data.new_subset()
        subset.label = "testing"
        item = self.get_item(subset)
        assert item.text(0) == "testing"
