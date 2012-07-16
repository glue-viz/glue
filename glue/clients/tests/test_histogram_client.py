import pytest

from mock import MagicMock

from ..histogram_client import HistogramClient

from ...tests import example_data
from ...core.data_collection import DataCollection
from ...core.exceptions import IncompatibleDataException
from ...core.hub import Hub

class TestException(Exception):
    pass

class TestHistogramClient(object):

    def setup_method(self, method):
        self.data = example_data.test_histogram_data()
        axes = MagicMock()
        self.collect = DataCollection(self.data)
        self.client = HistogramClient(self.collect, axes)
        self.axes = axes

    def draw_count(self):
        return self.axes.figure.canvas.draw.call_count

    def test_empty_on_creation(self):
        assert not self.client.layer_present(self.data)

    def test_add_layer(self):
        self.client.add_layer(self.data)
        assert self.client.layer_present(self.data)

    def test_add_invalid_layer_raises(self):
        self.collect.remove(self.data)
        with pytest.raises(IncompatibleDataException) as exc:
            self.client.add_layer(self.data)

    def test_add_subset_auto_adds_data(self):
        subset = self.data.new_subset()
        self.client.add_layer(subset)
        assert self.client.layer_present(self.data)
        assert self.client.layer_present(subset)

    def test_double_add_ignored(self):
        self.client.add_layer(self.data)
        mgr = self.client._managers[self.data]
        self.client.add_layer(self.data)
        assert self.client._managers[self.data] is mgr

    def test_add_data_auto_adds_subsets(self):
        self.client.add_layer(self.data)
        assert self.client.layer_present(self.data.edit_subset)

    def test_data_removal(self):
        self.client.add_layer(self.data)
        self.client.remove_layer(self.data)
        assert not (self.client.layer_present(self.data))

    def test_clear_patches_called_on_manager_pop(self):
        self.client.add_layer(self.data)
        mgr = self.client._managers[self.data]
        mgr.set_patches([MagicMock()])
        self.client.remove_layer(self.data)
        assert not mgr.has_patches()

    def test_data_removal_removes_subsets(self):
        self.client.add_layer(self.data)
        self.client.remove_layer(self.data)
        assert len(self.data.subsets) > 0

        for subset in self.data.subsets:
            assert not (self.client.layer_present(subset))

    def test_layer_updates_on_data_add(self):
        self.client.add_layer(self.data)
        assert self.draw_count() == 1 + len(self.data.subsets)

    def test_set_data_updates_active_data(self):
        self.client.add_layer(self.data)
        self.client.set_data(self.data)
        assert self.client._active_data is self.data

    def test_set_data_redraws(self):
        self.client.add_layer(self.data)
        ct0 = self.draw_count()
        self.client.set_data(self.data)
        assert self.draw_count() > ct0

    def test_set_data_auto_adds(self):
        self.client.set_data(self.data)
        assert (self.client.layer_present(self.data))

    def test_set_component_updates_component(self):
        self.client.add_layer(self.data)
        comp, = self.data.find_component_id('uniform')
        self.client.set_component(comp)
        assert self.client._component is comp

    def test_set_component_redraws(self):
        self.client.add_layer(self.data)
        comp, = self.data.find_component_id('uniform')
        ct0 = self.draw_count()
        self.client.set_component(comp)
        assert self.draw_count() > ct0

    def test_remove_not_present_ignored(self):
        self.client.remove_layer(self.data)

    def test_set_visible_external_data(self):
        self.client.set_layer_visible(None, False)

    def test_get_visible_external_data(self):
        assert not (self.client.is_layer_visible(None))

    def test_set_visible(self):
        self.client.add_layer(self.data)
        self.client.set_layer_visible(self.data, False)
        assert not (self.client.is_layer_visible(self.data))

    def test_draw_histogram_one_layer(self):
        self.client.add_layer(self.data)
        self.client.set_data(self.data)
        self.client.set_layer_visible(self.data, False)
        self.client.set_component(self.data.find_component_id('uniform')[0])

    def test_draw_histogram_subset_hidden(self):
        self.client.add_layer(self.data)
        self.client.set_data(self.data)
        self.client.set_layer_visible(self.data.edit_subset, False)
        self.client.set_component(self.data.find_component_id('uniform')[0])

    def test_draw_histogram_two_layers(self):
        self.client.add_layer(self.data)
        self.client.set_data(self.data)
        self.client.set_component(self.data.find_component_id('uniform')[0])

class TestCommunication(object):
    def setup_method(self, method):
        self.data = example_data.test_histogram_data()
        axes = MagicMock()
        self.collect = DataCollection()
        self.client = HistogramClient(self.collect, axes)
        self.axes = axes
        self.hub = Hub()
        self.connect()

    def draw_count(self):
        return self.axes.figure.canvas.draw.call_count

    def connect(self):
        self.client.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def test_ignore_data_add_message(self):
        self.collect.append(self.data)
        assert not (self.client.layer_present(self.data))

    def test_update_data_ignored_if_data_not_present(self):
        self.collect.append(self.data)
        ct0 = self.draw_count()
        self.data.style.color = 'blue'
        assert self.draw_count() == ct0

    def test_update_data_processed_if_data_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        ct0 = self.draw_count()
        self.data.style.color = 'blue'
        assert self.draw_count() > ct0

    def test_add_subset_ignored_if_data_not_present(self):
        self.collect.append(self.data)
        ct0 = self.draw_count()
        sub = self.data.new_subset()
        assert not (self.client.layer_present(sub))

    def test_add_subset_processed_if_data_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        assert (self.client.layer_present(sub))

    def test_update_subset_ignored_if_not_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        self.client.remove_layer(sub)
        ct0 = self.draw_count()
        sub.style.color='blue'
        assert self.draw_count() == ct0

    def test_update_subset_processed_if_present(self):
        self.collect.append(self.data)
        self.client.add_layer(self.data)
        sub = self.data.new_subset()
        ct0 = self.draw_count()
        sub.style.color='blue'
        assert self.draw_count() > ct0
