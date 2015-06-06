# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from ....external.qt.QtGui import QMainWindow
from ....external.qt.QtTest import QTest
from ....external.qt.QtCore import Qt

from mock import MagicMock, patch

from ..layer_tree_widget import (LayerTreeWidget, Clipboard,
                                 save_subset, PlotAction)

from ....tests import example_data
from .... import core


class TestLayerTree(object):

    """ Unit tests for the layer_tree_widget class """

    def setup_method(self, method):
        self.data = example_data.test_data()
        self.collect = core.data_collection.DataCollection(list(self.data))
        self.hub = self.collect.hub
        self.widget = LayerTreeWidget()
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)
        self.widget.setup(self.collect)
        for key, value in self.widget._actions.items():
            self.__setattr__("%s_action" % key, value)

    def teardown_method(self, method):
        self.win.close()

    def select_layers(self, *layers):
        self.widget.layerTree.set_selected_layers(layers)

    def remove_layer(self, layer):
        """ Remove a layer via the widget remove button """
        self.select_layers(layer)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)
        QTest.mouseRelease(self.widget.layerRemoveButton, Qt.LeftButton)

    def add_layer(self, layer=None):
        """ Add a layer through a hub message """
        layer = layer or core.Data()
        self.widget.data_collection.append(layer)
        return layer

    def layer_present(self, layer):
        """ Test that a layer exists in the data collection """
        return layer in self.collect or \
            getattr(layer, 'data', None) in self.collect

    def test_current_layer_method_correct(self):
        layer = self.add_layer()
        self.select_layers(layer)
        assert self.widget.current_layer() is layer

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

    def test_remove_subset_triggers_selection_changed(self):
        layer = self.add_layer()
        grp = self.collect.new_subset_group()
        mock = MagicMock()
        self.select_layers(grp)
        self.widget.layerTree.selection_changed.connect(mock)
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)
        QTest.mouseRelease(self.widget.layerRemoveButton, Qt.LeftButton)
        assert mock.call_count > 0

    def test_remove_subset_layer(self):
        """ Test that widget remove button works properly on subset groups"""
        layer = self.add_layer()
        grp = self.collect.new_subset_group()
        assert self.layer_present(grp)
        self.remove_layer(grp)
        assert not self.layer_present(grp)

    def test_empty_removal_does_nothing(self):
        """ Make sure widgets are only removed when selected """
        layer = self.add_layer()
        self.widget.layerTree.clearSelection()
        QTest.mousePress(self.widget.layerRemoveButton, Qt.LeftButton)
        assert self.layer_present(layer)

    @patch('glue.qt.widgets.layer_tree_widget.LinkEditor')
    def test_link_data(self, le):
        layer = self.add_layer()
        self.select_layers(layer)
        self.link_action.trigger()
        assert le.update_links.call_count == 1

    def test_new_subset_action(self):
        """ new action creates a new subset group """
        layer = self.add_layer()
        self.new_action.trigger()
        assert len(self.collect.subset_groups) == 1

    def test_maskify_action(self):
        d = core.Data(x=[1, 2, 3])
        s = d.new_subset()

        selected = MagicMock()
        self.maskify_action.selected_layers = selected
        selected.return_value = [s]

        self.maskify_action.trigger()
        assert isinstance(s.subset_state, core.subset.MaskSubsetState)

    def test_copy_paste_subset_action(self):
        layer = self.add_layer()
        grp = self.collect.new_subset_group()
        self.select_layers(grp)
        self.copy_action.trigger()
        grp2 = self.collect.new_subset_group()
        self.select_layers(grp2)
        state0 = grp2.subset_state
        self.paste_action.trigger()
        assert grp2.subset_state is not state0

    def setup_two_subset_selection(self):
        layer = self.add_layer()
        g1 = self.collect.new_subset_group()
        g2 = self.collect.new_subset_group()
        self.select_layers(g1, g2)
        return layer

    def test_invert(self):
        layer = self.add_layer()
        sub = self.collect.new_subset_group()
        self.select_layers(sub)
        self.invert_action.trigger()
        assert isinstance(sub.subset_state, core.subset.InvertState)

    def test_actions_enabled_single_subset_group_selection(self):
        Clipboard().contents = None
        layer = self.add_layer()
        grp = self.collect.new_subset_group()
        self.select_layers(grp)

        assert self.new_action.isEnabled()
        assert self.copy_action.isEnabled()
        assert not self.paste_action.isEnabled()
        assert self.invert_action.isEnabled()
        assert self.clear_action.isEnabled()

    def test_actions_enabled_single_data_selection(self):
        layer = self.add_layer()
        self.select_layers(layer)

        assert self.new_action.isEnabled()
        assert not self.copy_action.isEnabled()
        assert not self.paste_action.isEnabled()
        assert not self.invert_action.isEnabled()
        assert not self.clear_action.isEnabled()

    def test_actions_enabled_multi_subset_group_selection(self):
        layer = self.setup_two_subset_selection()
        assert self.new_action.isEnabled()
        assert not self.copy_action.isEnabled()
        assert not self.paste_action.isEnabled()
        assert not self.invert_action.isEnabled()
        assert not self.clear_action.isEnabled()

    def test_checkable_toggle(self):
        self.widget.set_checkable(True)
        assert self.widget.is_checkable()
        self.widget.set_checkable(False)
        assert not self.widget.is_checkable()

    def test_load_data(self):
        pth = 'glue.qt.widgets.layer_tree_widget.qtutil.data_wizard'
        with patch(pth) as wizard:
            d = core.Data(x=[1])
            assert not self.layer_present(d)
            wizard.return_value = [d]
            self.widget._load_data()
            assert self.layer_present(d)

    def test_clear_subset_group(self):
        layer = self.add_layer()
        sub = self.collect.new_subset_group()
        self.select_layers(sub)
        dummy_state = MagicMock()
        sub.subset_state = dummy_state
        self.clear_action.trigger()
        assert sub.subset_state is not dummy_state

    def test_single_selection_updates_editable(self):
        self.widget.bind_selection_to_edit_subset()
        layer = self.add_layer()
        grp1 = self.collect.new_subset_group()
        grp2 = self.collect.new_subset_group()
        assert layer.edit_subset[0].group is not grp1
        self.select_layers(grp1)
        assert layer.edit_subset[0].group is grp1

    def test_multi_selection_updates_editable(self):
        """Selection disables edit_subset for all other data"""
        self.widget.bind_selection_to_edit_subset()
        layer = self.add_layer()
        layer2 = self.add_layer()
        grps = [self.collect.new_subset_group() for _ in range(3)]
        self.select_layers(*grps[:2])
        selected = [s.group for s in layer.edit_subset + layer2.edit_subset]
        assert grps[0] in selected
        assert grps[1] in selected
        assert grps[2] not in selected

    def test_selection_updates_on_data_add(self):
        layer = self.add_layer()
        assert self.widget.selected_layers() == [layer]

    def test_selection_updates_on_subset_group_add(self):
        layer = self.add_layer()
        grp = self.collect.new_subset_group()
        assert self.widget.selected_layers() == [grp]

    def test_save_subset(self):
        subset = MagicMock(core.Subset)
        with patch('glue.qt.widgets.layer_tree_widget.QFileDialog') as d:
            d.getSaveFileName.return_value = ('test.fits', None)
            save_subset(subset)
        subset.write_mask.assert_called_once_with('test.fits')

    def test_save_subset_cancel(self):
        subset = MagicMock(core.Subset)
        with patch('glue.qt.widgets.layer_tree_widget.QFileDialog') as d:
            d.getSaveFileName.return_value = ('', '')
            save_subset(subset)
        assert subset.write_mask.call_count == 0

    def test_plot_action(self):
        # regression test for #364

        app = MagicMock()
        pa = PlotAction(self.widget, app)

        layer = self.add_layer()
        grp = self.collect.new_subset_group()

        self.select_layers(grp)
        assert not pa.isEnabled()

        self.select_layers(layer)
        assert pa.isEnabled()
