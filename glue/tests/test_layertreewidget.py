import unittest
from threading import Thread

from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QItemSelectionModel

from mock import MagicMock, patch

import glue
from glue.qt.layertreewidget import LayerTreeWidget
from glue.subset import OrState, AndState, XorState, InvertState

class TestLayerTree(unittest.TestCase):
    """ Unit tests for the layertreewidget class """
    def setUp(self):
        import sys

        self.app = QApplication(sys.argv)
        self.data = glue.example_data.test_data()
        self.hub = glue.Hub()
        self.collect = glue.DataCollection(list(self.data))
        self.widget = LayerTreeWidget()
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)

    def tearDown(self):
        self.win.close()
        del self.win
        del self.app

    def add_layer_via_method(self):
        """ Add first layer by invoking method"""
        self.widget.add_layer(self.data[0])
        return self.data[0]

    def remove_layer_via_button(self, layer):
        """ Remove a layer via the widget remove button """
        widget_item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(widget_item)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)

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
        self.assertIs(self.widget.current_layer(), layer)

    def test_current_layer_null_on_creation(self):
        self.assertIs(None, self.widget.current_layer())

    def test_add_layer_method(self):
        """ Test that a layer exists once added """
        self.assertFalse(self.layer_present(self.data[0]))
        layer = self.add_layer_via_method()
        self.assertTrue(self.layer_present(layer))

    def test_remove_layer(self):
        """ Test that widget remove button works properly """
        layer = self.add_layer_via_method()
        self.remove_layer_via_button(layer)
        self.assertFalse(self.layer_present(layer))
        self.assertFalse(layer in self.widget.data_collection)

    def test_remove_layer_ignored_if_edit_subset(self):
        layer = self.add_layer_via_method()
        self.remove_layer_via_button(layer.edit_subset)
        self.assertTrue(self.layer_present(layer.edit_subset))

    def test_empty_removal_does_nothing(self):
        """ Make sure widgets are only removed when selected """
        layer = self.add_layer_via_method()
        widget_item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(widget_item)
        self.widget.layerTree.setItemSelected(widget_item, False)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)
        self.assertTrue(self.layer_present(layer))

    def test_add_via_hub_message(self):
        """ Test that hub messages properly add layers """
        layer = self.add_layer_via_hub()
        self.assertTrue(self.layer_present(layer))

    def test_remove_via_hub_message(self):
        """ Test that hub messages properly remove layers """
        layer = self.add_layer_via_hub()
        self.collect.remove(layer)
        self.assertFalse(self.layer_present(layer))

    def test_update_via_hub_message(self):
        """ Test that hub messages properly update layers """
        layer = self.add_layer_via_hub()
        layer.style.markersize = 999
        assert layer in self.widget
        widget_item = self.widget[layer]
        self.assertTrue(str(widget_item.text(3)) == "999")

    def test_data_added_with_subset_add(self):
        """ Test that subset layers are added properly """
        layer = self.data[0]
        sub = glue.Subset(layer)
        self.widget.add_layer(sub)
        self.assertTrue(self.layer_present(sub))
        self.assertTrue(self.layer_present(layer))

    def test_subsets_added_with_data(self):
        layer = self.add_layer_via_method()
        self.assertTrue(self.layer_present(layer.subsets[0]))

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
        self.assertEquals(len(layer.subsets), 2)

    def test_duplicate_subset_action(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer.subsets[0]]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._duplicate_action.trigger()
        self.assertEquals(len(layer.subsets), 2)

    def test_copy_paste_subset_action(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer.subsets[0]]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._copy_action.trigger()
        self.assertTrue(self.widget._clipboard is not None)
        sub = layer.new_subset()
        self.widget.add_layer(sub)
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        state0 = sub.subset_state
        self.widget._paste_action.trigger()
        self.assertTrue(sub.subset_state is not state0)

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
        self.assertEquals(len(self.widget.layerTree.selectedItems()), 2)
        return layer

    def test_or_combine(self):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        self.widget._or_action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        self.assertTrue(len(diff) == 1)
        self.assertTrue(isinstance(diff[0].subset_state, OrState))

    def test_and_combine(self):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        self.widget._and_action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        self.assertTrue(len(diff) == 1)
        self.assertTrue(isinstance(diff[0].subset_state, AndState))

    def test_invert(self):
        layer = self.add_layer_via_method()
        sub = layer.edit_subset
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        self.widget._invert_action.trigger()
        self.assertTrue(isinstance(sub.subset_state, InvertState))

    def test_xor_combine(self):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        self.widget._xor_action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        self.assertTrue(len(diff) == 1)
        self.assertTrue(isinstance(diff[0].subset_state, XorState))


    def test_actions_enabled_single_subset_selection(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer.edit_subset]
        self.widget.layerTree.setCurrentItem(item)

        self.assertFalse(self.widget._or_action.isEnabled())
        self.assertFalse(self.widget._and_action.isEnabled())
        self.assertFalse(self.widget._xor_action.isEnabled())
        self.assertTrue(self.widget._new_action.isEnabled())
        self.assertTrue(self.widget._copy_action.isEnabled())
        self.assertTrue(self.widget._duplicate_action.isEnabled())
        self.assertFalse(self.widget._paste_action.isEnabled())
        self.assertTrue(self.widget._invert_action.isEnabled())
        self.assertTrue(self.widget._clear_action.isEnabled())

    def test_actions_enabled_single_data_selection(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)

        self.assertFalse(self.widget._or_action.isEnabled())
        self.assertFalse(self.widget._and_action.isEnabled())
        self.assertFalse(self.widget._xor_action.isEnabled())
        self.assertTrue(self.widget._new_action.isEnabled())
        self.assertFalse(self.widget._copy_action.isEnabled())
        self.assertFalse(self.widget._duplicate_action.isEnabled())
        self.assertFalse(self.widget._paste_action.isEnabled())
        self.assertFalse(self.widget._invert_action.isEnabled())
        self.assertFalse(self.widget._clear_action.isEnabled())

    def test_actions_enabled_multi_subset_selection(self):
        layer = self.setup_two_subset_selection()
        self.assertTrue(self.widget._or_action.isEnabled())
        self.assertTrue(self.widget._and_action.isEnabled())
        self.assertTrue(self.widget._xor_action.isEnabled())
        self.assertFalse(self.widget._new_action.isEnabled())
        self.assertFalse(self.widget._copy_action.isEnabled())
        self.assertFalse(self.widget._duplicate_action.isEnabled())
        self.assertFalse(self.widget._paste_action.isEnabled())
        self.assertTrue(self.widget._invert_action.isEnabled())
        self.assertFalse(self.widget._clear_action.isEnabled())

    def test_remove_all_layers(self):
        for d in self.collect:
            self.widget.add_layer(d)

        self.widget.remove_all_layers()

        for d in self.collect:
            self.assertFalse(self.layer_present(d))

    def test_checkable_toggle(self):
        self.widget.set_checkable(True)
        self.assertTrue(self.widget.is_checkable())
        self.widget.set_checkable(False)
        self.assertFalse(self.widget.is_checkable())

    def test_remove_layer_keeps_layer_in_collection(self):
        layer = self.add_layer_via_method()
        self.widget.remove_layer(layer)
        self.assertTrue(layer in self.widget.data_collection)

    def test_remove_and_delete_removes_from_collection(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        self.widget.remove_and_delete_selected()
        self.assertFalse(self.layer_present(layer))
        self.assertFalse(layer in self.widget.data_collection)

    def test_set_data_collection(self):
        layer = self.add_layer_via_method()
        dc = glue.DataCollection()
        self.widget.data_collection = dc
        self.assertTrue(self.widget.data_collection is dc)
        self.assertFalse(self.layer_present(layer))

    def test_edit_layer_label(self):
        with patch('glue.qt.layertreewidget.qtutil') as util:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 0)
            util.edit_layer_label = MagicMock()
            self.widget.edit_current_layer()
            util.edit_layer_label.assert_called_once_with(layer)

    def test_edit_layer_color(self):
        with patch('glue.qt.layertreewidget.qtutil') as util:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 1)
            self.widget.edit_current_layer()
            util.edit_layer_color.assert_called_once_with(layer)

    def test_edit_layer_symbol(self):
        with patch('glue.qt.layertreewidget.qtutil') as util:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 2)
            self.widget.edit_current_layer()
            util.edit_layer_symbol.assert_called_once_with(layer)

    def test_edit_layer_point_size(self):
        with patch('glue.qt.layertreewidget.qtutil') as util:
            layer = self.add_layer_via_method()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 3)
            self.widget.edit_current_layer()
            util.edit_layer_point_size.assert_called_once_with(layer)

    def test_add_data_creates_default_label(self):
        layer = self.data[0]
        layer.style.label = None
        self.widget.add_layer(layer)
        self.assertTrue(layer in self.widget)
        self.assertEquals(layer.style.label, "Data 0")

    def test_load_data(self):
        with patch('glue.qt.layertreewidget.qtutil') as util:
            util.data_wizard.return_value = self.data[0]
            self.widget._load_data()
            self.assertTrue(self.layer_present(self.data[0]))

    def test_load_data_doesnt_double_add(self):
        """bugfix"""
        with patch('glue.qt.layertreewidget.qtutil') as util:
            util.data_wizard.return_value = self.data[0]
            self.assertEquals(self.widget.layerTree.topLevelItemCount(), 0)
            self.widget._load_data()
            self.assertEquals(self.widget.layerTree.topLevelItemCount(), 1)

    def test_sync_external_layer_ignored(self):
        self.assertFalse(self.data[0] in self.widget)
        self.widget.sync_layer(self.data[0])

    def test_clear_subset(self):
        layer = self.add_layer_via_method()
        sub = layer.edit_subset
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        dummy_state = MagicMock()
        sub.subset_state = dummy_state
        self.widget._clear_subset()
        self.assertFalse(sub.subset_state == dummy_state)

if __name__ == "__main__":
    unittest.main()