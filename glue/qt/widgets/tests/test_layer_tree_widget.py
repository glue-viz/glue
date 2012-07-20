from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QItemSelectionModel

from mock import MagicMock, patch
import pytest

from ..layer_tree_widget import LayerTreeWidget, Clipboard

from ....tests import example_data
from .... import core

def setup_module(module):
    module.app = QApplication([''])

def teardown_module(module):
    module.app.exit()
    del module.app

class TestLayerTree(object):
    """ Unit tests for the layer_tree_widget class """

    def setup_method(self, method):
        self.data = example_data.test_data()
        self.hub = core.hub.Hub()
        self.collect = core.data_collection.DataCollection(list(self.data))
        self.widget = LayerTreeWidget()
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)
        self.widget.setup(self.collect, self.hub)
        for key, value in self.widget._actions.items():
            self.__setattr__("%s_action" % key, value)

    def teardown_method(self, method):
        self.win.close()
        del self.win

    def remove_layer(self, layer):
        """ Remove a layer via the widget remove button """
        widget_item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(widget_item)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)

    def add_layer(self, layer=None):
        """ Add a layer through a hub message """
        layer = layer or core.Data()
        self.widget.data_collection.append(layer)
        assert layer in self.widget.data_collection
        assert layer in self.widget
        return layer

    def layer_present(self, layer):
        """ Test that a layer exists in the widget """
        return self.widget.is_layer_present(layer) and \
            layer.data in self.widget.data_collection

    def test_current_layer_method_correct(self):
        layer = self.add_layer()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        assert self.widget.current_layer() is layer

    def test_current_layer_null_on_creation(self):
        assert self.widget.current_layer() is None

    def test_add(self):
        """ Test that a layer exists in widget once added """
        data = core.Data()
        assert not self.layer_present(data)
        self.add_layer(data)
        assert self.layer_present(data)

    def test_remove_layer(self):
        """ Test that widget remove button works properly """
        layer = self.add_layer()
        self.remove_layer(layer)
        assert not self.layer_present(layer)
        assert not layer in self.widget.data_collection

    def test_remove_subset_layer(self):
        """ Test that widget remove button works properly on subsets"""
        layer = self.add_layer()
        subset = layer.new_subset()
        assert self.layer_present(subset)
        self.remove_layer(subset)
        assert not self.layer_present(subset)

    def test_remove_layer_ignored_if_edit_subset(self):
        layer = self.add_layer()
        self.remove_layer(layer.edit_subset)
        assert self.layer_present(layer.edit_subset)

    def test_empty_removal_does_nothing(self):
        """ Make sure widgets are only removed when selected """
        layer = self.add_layer()
        widget_item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(widget_item)
        self.widget.layerTree.setItemSelected(widget_item, False)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)
        assert self.layer_present(layer)

    @pytest.skip("Triggering seg faults for some reason")
    def test_check_signal(self):
        layer = self.add_layer()
        sc = MagicMock()
        self.widget._layer_check_changed.connect(sc.catch)
        item = self.widget[layer]
        item.setCheckState(0, Qt.Unchecked)
        sc.catch.assert_called_once_with(layer, False)

    def test_new_subset_action(self):
        """ new action creates a new subset """
        layer = self.add_layer()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        self.new_action.trigger()
        assert len(layer.subsets) == 2

    def test_duplicate_subset_action(self):
        """ duplicate action creates duplicate subset """
        layer = self.add_layer()
        item = self.widget[layer.subsets[0]]
        self.widget.layerTree.setCurrentItem(item)
        self.duplicate_action.trigger()
        assert len(layer.subsets) == 2

    def test_copy_paste_subset_action(self):
        layer = self.add_layer()
        item = self.widget[layer.subsets[0]]
        self.widget.layerTree.setCurrentItem(item)
        self.copy_action.trigger()
        sub = layer.new_subset()
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        state0 = sub.subset_state
        self.paste_action.trigger()
        assert sub.subset_state is not state0

    def setup_two_subset_selection(self):
        layer = self.add_layer()
        s2 = layer.new_subset()
        item1 = self.widget[layer.edit_subset]
        item2 = self.widget[s2]
        self.widget.layerTree.setCurrentItem(item1, 0,
                                             QItemSelectionModel.Toggle)
        self.widget.layerTree.setCurrentItem(item2, 0,
                                             QItemSelectionModel.Toggle)
        assert len(self.widget.layerTree.selectedItems()) == 2
        return layer

    def _test_combine(self, action, state):
        layer = self.setup_two_subset_selection()
        old_subsets = set(layer.subsets)
        action.trigger()
        new_subsets = set(layer.subsets)
        diff = list(old_subsets ^ new_subsets)
        assert len(diff) == 1
        assert isinstance(diff[0].subset_state, state)

    def test_or_combine(self):
        self._test_combine(self.or_action, core.subset.OrState)

    def test_and_combine(self):
        self._test_combine(self.and_action, core.subset.AndState)

    def test_xor_combine(self):
        self._test_combine(self.xor_action, core.subset.XorState)

    def test_invert(self):
        layer = self.add_layer()
        sub = layer.edit_subset
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        self.invert_action.trigger()
        assert isinstance(sub.subset_state, core.subset.InvertState)


    def test_actions_enabled_single_subset_selection(self):
        Clipboard().contents = None
        layer = self.add_layer()
        item = self.widget[layer.edit_subset]
        self.widget.layerTree.setCurrentItem(item)

        assert not self.or_action.isEnabled()
        assert not self.and_action.isEnabled()
        assert not self.xor_action.isEnabled()
        assert self.new_action.isEnabled()
        assert self.copy_action.isEnabled()
        assert self.duplicate_action.isEnabled()
        assert not self.paste_action.isEnabled()
        assert self.invert_action.isEnabled()
        assert self.clear_action.isEnabled()

    def test_actions_enabled_single_data_selection(self):
        layer = self.add_layer()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)

        assert not self.or_action.isEnabled()
        assert not self.and_action.isEnabled()
        assert not self.xor_action.isEnabled()
        assert self.new_action.isEnabled()
        assert not self.copy_action.isEnabled()
        assert not self.duplicate_action.isEnabled()
        assert not self.paste_action.isEnabled()
        assert not self.invert_action.isEnabled()
        assert not self.clear_action.isEnabled()

    def test_actions_enabled_multi_subset_selection(self):
        layer = self.setup_two_subset_selection()
        assert self.or_action.isEnabled()
        assert self.and_action.isEnabled()
        assert self.xor_action.isEnabled()
        assert not self.new_action.isEnabled()
        assert not self.copy_action.isEnabled()
        assert not self.duplicate_action.isEnabled()
        assert not self.paste_action.isEnabled()
        assert not self.invert_action.isEnabled()
        assert not self.clear_action.isEnabled()

    def test_checkable_toggle(self):
        self.widget.set_checkable(True)
        assert self.widget.is_checkable()
        self.widget.set_checkable(False)
        assert not self.widget.is_checkable()

    def test_edit_layer_label(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_label'
        with patch(pth) as edit_layer_label:
            layer = self.add_layer()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 0)
            self.widget.edit_current_layer()
            edit_layer_label.assert_called_once_with(layer)

    def test_edit_layer_color(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_color'
        with patch(pth) as edit_layer_color:
            layer = self.add_layer()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 1)
            self.widget.edit_current_layer()
            edit_layer_color.assert_called_once_with(layer)

    def test_edit_layer_symbol(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_symbol'
        with patch(pth) as edit_layer_symbol:
            layer = self.add_layer()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 2)
            self.widget.edit_current_layer()
            edit_layer_symbol.assert_called_once_with(layer)

    def test_edit_layer_point_size(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.edit_layer_point_size'
        with patch(pth) as edit_layer_point_size:
            layer = self.add_layer()
            item = self.widget[layer]
            self.widget.layerTree.setCurrentItem(item, 3)
            self.widget.edit_current_layer()
            edit_layer_point_size.assert_called_once_with(layer)

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
            wizard.return_value = core.Data()
            ct0 = self.widget.layerTree.topLevelItemCount()
            self.widget._load_data()
            assert self.widget.layerTree.topLevelItemCount() == ct0+1

    def test_clear_subset(self):
        layer = self.add_layer()
        sub = layer.edit_subset
        item = self.widget[sub]
        self.widget.layerTree.setCurrentItem(item)
        dummy_state = MagicMock()
        sub.subset_state = dummy_state
        self.clear_action.trigger()
        assert not sub.subset_state == dummy_state

