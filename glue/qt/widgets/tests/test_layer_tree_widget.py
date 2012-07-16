from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QItemSelectionModel

from mock import MagicMock, patch

from ..layer_tree_widget import LayerTreeWidget

from ....tests import example_data
from .... import core


class TestLayerTree(object):
    """ Unit tests for the layer_tree_widget class """

    def setup_method(self, method):
        import sys

        self.app = QApplication(sys.argv)
        self.data = example_data.test_data()
        self.hub = core.hub.Hub()
        self.collect = core.data_collection.DataCollection(list(self.data))
        self.widget = LayerTreeWidget()
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)

    def tearDown(self):
        self.win.close()
        del self.win
        del self.app

    def add_layer_via_method(self, layer=None):
        """ Add a layer (default = first data set) by invoking method"""
        layer = layer or self.data[0]
        self.widget.add_layer(layer)
        return layer

    def remove_layer_via_button(self, layer):
        """ Remove a layer via the widget remove button """
        widget_item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(widget_item)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)

    def remove_layer_via_method(self, layer):
        self.widget.remove_and_delete_layer(layer)

    def add_layer_via_hub(self):
        """ Add a layer through a hub message """
        self.widget.data_collection = self.collect
        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)
        layer = self.data[0]
        self.collect.append(layer)
        return layer

    def layer_present(self, layer):
        """ Test that a layer exists in the widget """
        return self.widget.is_layer_present(layer) and \
            layer.data in self.widget.data_collection

    def test_current_layer_method_correct(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        assert self.widget.current_layer() is layer

    def test_current_layer_null_on_creation(self):
        assert self.widget.current_layer() is None

    def test_add_layer_method(self):
        """ Test that a layer exists once added """
        assert not self.layer_present(self.data[0])
        layer = self.add_layer_via_method()
        assert self.layer_present(layer)

    def test_remove_layer(self):
        """ Test that widget remove button works properly """
        layer = self.add_layer_via_method()
        self.remove_layer_via_button(layer)
        assert not self.layer_present(layer)
        assert not layer in self.widget.data_collection

    def test_remove_subset_layer(self):
        """ Test that widget remove button works properly on subsets"""
        layer = self.add_layer_via_method()
        subset = layer.new_subset()
        self.add_layer_via_method(subset)
        self.remove_layer_via_button(subset)
        assert not self.layer_present(subset)

    def test_remove_layer_ignored_if_edit_subset(self):
        layer = self.add_layer_via_method()
        self.remove_layer_via_button(layer.edit_subset)
        assert self.layer_present(layer.edit_subset)

    def test_empty_removal_does_nothing(self):
        """ Make sure widgets are only removed when selected """
        layer = self.add_layer_via_method()
        widget_item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(widget_item)
        self.widget.layerTree.setItemSelected(widget_item, False)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)
        assert self.layer_present(layer)

    def test_add_via_hub_message(self):
        """ Test that hub messages properly add layers """
        layer = self.add_layer_via_hub()
        assert self.layer_present(layer)

    def test_remove_via_hub_message(self):
        """ Test that hub messages properly remove layers """
        layer = self.add_layer_via_hub()
        self.collect.remove(layer)
        assert not self.layer_present(layer)

    def test_update_via_hub_message(self):
        """ Test that hub messages properly update layers """
        layer = self.add_layer_via_hub()
        layer.style.markersize = 999
        assert layer in self.widget
        widget_item = self.widget[layer]
        assert str(widget_item.text(3)) == "999"

    def test_data_added_with_subset_add(self):
        """ Test that subset layers are added properly """
        layer = self.data[0]
        sub = core.subset.Subset(layer)
        self.widget.add_layer(sub)
        assert self.layer_present(sub)
        assert self.layer_present(layer)

    def test_subsets_added_with_data(self):
        layer = self.add_layer_via_method()
        assert self.layer_present(layer.subsets[0])

    def test_check_signal(self):
        layer = self.add_layer_via_method()
        sc = MagicMock()
        self.widget._layer_check_changed.connect(sc.catch)
        item = self.widget[layer]
        item.setCheckState(0, Qt.Unchecked)
        sc.catch.assert_called_once_with(layer, False)

    def test_new_subset_action(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._new_action.trigger()
        assert len(layer.subsets) == 2

    def test_duplicate_subset_action(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer.subsets[0]]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._duplicate_action.trigger()
        assert len(layer.subsets) == 2

    def test_copy_paste_subset_action(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer.subsets[0]]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._copy_action.trigger()
        assert self.widget._clipboard is not None
        sub = layer.new_subset()
        self.widget.add_layer(sub)
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        state0 = sub.subset_state
        self.widget._paste_action.trigger()
        assert sub.subset_state is not state0

    def setup_two_subset_selection(self):
        layer = self.add_layer_via_method()
        s2 = layer.new_subset()
        self.widget.add_layer(s2)
        item1 = self.widget[layer.edit_subset]
        item2 = self.widget[s2]
        self.widget.layerTree.setCurrentItem(item1, 0,
                                             QItemSelectionModel.Toggle)
        self.widget.layerTree.setCurrentItem(item2, 0,
                                             QItemSelectionModel.Toggle)
        assert len(self.widget.layerTree.selectedItems()) == 2
        return layer

    def test_or_combine(self):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        self.widget._or_action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        assert len(diff) == 1
        assert isinstance(diff[0].subset_state, core.subset.OrState)

    def test_and_combine(self):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        self.widget._and_action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        assert len(diff) == 1
        assert isinstance(diff[0].subset_state, core.subset.AndState)

    def test_invert(self):
        layer = self.add_layer_via_method()
        sub = layer.edit_subset
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._invert_action.trigger()
        assert isinstance(sub.subset_state, core.subset.InvertState)

    def test_xor_combine(self):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        self.widget._xor_action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        assert len(diff) == 1
        assert isinstance(diff[0].subset_state, core.subset.XorState)

    def test_actions_enabled_single_subset_selection(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer.edit_subset]
        self.widget.layerTree.setCurrentItem(item)

        assert not self.widget._or_action.isEnabled()
        assert not self.widget._and_action.isEnabled()
        assert not self.widget._xor_action.isEnabled()
        assert self.widget._new_action.isEnabled()
        assert self.widget._copy_action.isEnabled()
        assert self.widget._duplicate_action.isEnabled()
        assert not self.widget._paste_action.isEnabled()
        assert self.widget._invert_action.isEnabled()
        assert self.widget._clear_action.isEnabled()

    def test_actions_enabled_single_data_selection(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)

        assert not self.widget._or_action.isEnabled()
        assert not self.widget._and_action.isEnabled()
        assert not self.widget._xor_action.isEnabled()
        assert self.widget._new_action.isEnabled()
        assert not self.widget._copy_action.isEnabled()
        assert not self.widget._duplicate_action.isEnabled()
        assert not self.widget._paste_action.isEnabled()
        assert not self.widget._invert_action.isEnabled()
        assert not self.widget._clear_action.isEnabled()

    def test_actions_enabled_multi_subset_selection(self):
        layer = self.setup_two_subset_selection()
        assert self.widget._or_action.isEnabled()
        assert self.widget._and_action.isEnabled()
        assert self.widget._xor_action.isEnabled()
        assert not self.widget._new_action.isEnabled()
        assert not self.widget._copy_action.isEnabled()
        assert not self.widget._duplicate_action.isEnabled()
        assert not self.widget._paste_action.isEnabled()
        assert self.widget._invert_action.isEnabled()
        assert not self.widget._clear_action.isEnabled()

    def test_remove_all_layers(self):
        for d in self.collect:
            self.widget.add_layer(d)

        self.widget.remove_all_layers()

        for d in self.collect:
            assert not self.layer_present(d)

    def test_checkable_toggle(self):
        self.widget.set_checkable(True)
        assert self.widget.is_checkable()
        self.widget.set_checkable(False)
        assert not self.widget.is_checkable()

    def test_remove_layer_keeps_layer_in_collection(self):
        layer = self.add_layer_via_method()
        self.widget.remove_layer(layer)
        assert layer in self.widget.data_collection

    def test_remove_and_delete_removes_from_collection(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        self.widget.remove_and_delete_selected()
        assert not self.layer_present(layer)
        assert not layer in self.widget.data_collection

    def test_set_data_collection(self):
        layer = self.add_layer_via_method()
        dc = core.data_collection.DataCollection()
        self.widget.data_collection = dc
        assert self.widget.data_collection is dc
        assert not self.layer_present(layer)

    def test_edit_layer_label(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_label'
        with patch(pth) as edit_layer_label:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 0)
            self.widget.edit_current_layer()
            edit_layer_label.assert_called_once_with(layer)

    def test_edit_layer_color(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_color'
        with patch(pth) as edit_layer_color:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 1)
            self.widget.edit_current_layer()
            edit_layer_color.assert_called_once_with(layer)

    def test_edit_layer_symbol(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_symbol'
        with patch(pth) as edit_layer_symbol:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 2)
            self.widget.edit_current_layer()
            edit_layer_symbol.assert_called_once_with(layer)

    def test_edit_layer_point_size(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_point_size'
        with patch(pth) as edit_layer_point_size:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 3)
            self.widget.edit_current_layer()
            edit_layer_point_size.assert_called_once_with(layer)

    def test_add_data_creates_default_label(self):
        layer = self.data[0]
        layer.style.label = None
        self.widget.add_layer(layer)
        assert layer in self.widget
        assert layer.style.label == "Data 0"

    def test_load_data(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.data_wizard'
        with patch(pth) as wizard:
            wizard.return_value = self.data[0]
            self.widget._load_data()
            assert self.layer_present(self.data[0])

    def test_load_data_doesnt_double_add(self):
        """bugfix"""
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.data_wizard'
        with patch(pth) as wizard:
            wizard.return_value = self.data[0]
            assert self.widget.layerTree.topLevelItemCount() == 0
            self.widget._load_data()
            assert self.widget.layerTree.topLevelItemCount() == 1

    def test_sync_external_layer_ignored(self):
        assert not self.data[0] in self.widget
        self.widget.sync_layer(self.data[0])

    def test_clear_subset(self):
        layer = self.add_layer_via_method()
        sub = layer.edit_subset
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        dummy_state = MagicMock()
        sub.subset_state = dummy_state
        self.widget._clear_subset()
        assert not sub.subset_state == dummy_state

    def test_remove_message_global(self):
        """RemoveData messages should be processed, regardless of origin"""
        w = LayerTreeWidget()
        data = core.data.Data()
        hub = core.hub.Hub(w)
        dc = core.data_collection.DataCollection()

        w.add_layer(data)

        msg = core.message.DataCollectionDeleteMessage(dc, data)
        hub.broadcast(msg)

        assert data not in w

