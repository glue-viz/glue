import unittest

import numpy as np
from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

import glue
from glue.qt.widgets.scatter_widget import ScatterWidget

import example_data

class TestScatterWidget(unittest.TestCase):

    def setUp(self):
        import sys
        self.app = QApplication(sys.argv)
        self.hub = glue.core.hub.Hub()
        self.data = example_data.test_data()
        self.collect = glue.core.data_collection.DataCollection()
        self.widget = ScatterWidget(self.collect)
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)
        self.connect_to_hub()
        self.win.show()

    def tearDown(self):
        self.win.close()
        del self.win
        del self.app

    def connect_to_hub(self):
        self.widget.register_to_hub(self.hub)
        self.collect.register_to_hub(self.hub)

    def add_layer_via_hub(self):
        layer = self.data[0]
        self.collect.append(layer)
        return layer

    def add_layer_via_method(self):
        layer = self.data[0]
        self.collect.append(layer)
        self.widget.add_layer(layer)
        return layer

    def plot_data(self, layer):
        """ Return the data bounds for a given layer (data or subset)
        Output format: [xmin, xmax], [ymin, ymax]
        """
        client = self.widget.client
        data = client.managers[layer].get_data()
        xmin = data[:, 0].min()
        xmax = data[:, 0].max()
        ymin = data[:, 1].min()
        ymax = data[:, 1].max()
        return [xmin, xmax], [ymin, ymax]


    def plot_limits(self):
        """ Return the plot limits
        Output format [xmin, xmax], [ymin, ymax]
        """
        ax = self.widget.client.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return xlim, ylim
        return

    def assert_layer_inside_limits(self, layer):
        """Assert that points of a layer are within plot limits """
        xydata = self.plot_data(layer)
        xylimits = self.plot_limits()

        self.assertGreaterEqual(xydata[0][0], xylimits[0][0])
        self.assertGreaterEqual(xydata[1][0], xylimits[1][0])
        self.assertLessEqual(xydata[0][1], xylimits[0][1])
        self.assertLessEqual(xydata[1][1], xylimits[1][1])

    def set_layer_checkbox(self, layer, state):
        item = self.widget.ui.layerTree[layer]
        item.setCheckState(0, state)

    def is_layer_present(self, layer):
        if not self.widget.ui.layerTree.is_layer_present(layer):
            return False
        return self.widget.client.is_layer_present(layer)

    def is_layer_visible(self, layer):
        return self.widget.client.is_visible(layer)

    def test_rescaled_on_init(self):
        layer = self.add_layer_via_method()
        self.assert_layer_inside_limits(layer)

    def test_hub_data_add_is_ignored(self):
        layer = self.add_layer_via_hub()
        assert not self.widget.ui.layerTree.is_layer_present(layer)
        assert not self.widget.client.is_layer_present(layer)

    def test_valid_add_data_via_method(self):
        layer = self.add_layer_via_method()
        assert self.is_layer_present(layer)

    def test_add_first_data_updates_combos(self):
        layer = self.add_layer_via_method()
        xatt = str(self.widget.ui.xAxisComboBox.currentText())
        yatt = str(self.widget.ui.yAxisComboBox.currentText())
        self.assertIsNotNone(xatt)
        self.assertIsNotNone(yatt)

    def test_flip_x(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.xFlipCheckBox, Qt.LeftButton)
        assert self.widget.client.is_xflip()
        QTest.mouseClick(self.widget.ui.xFlipCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_xflip()

    def test_flip_y(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.yFlipCheckBox, Qt.LeftButton)
        assert self.widget.client.is_yflip()
        QTest.mouseClick(self.widget.ui.yFlipCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_yflip()

    def test_log_x(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.xLogCheckBox, Qt.LeftButton)
        assert self.widget.client.is_xlog()
        QTest.mouseClick(self.widget.ui.xLogCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_xlog()

    def test_log_y(self):
        QTest.mouseClick(self.widget.ui.yLogCheckBox, Qt.LeftButton)
        assert self.widget.client.is_ylog()
        QTest.mouseClick(self.widget.ui.yLogCheckBox, Qt.LeftButton)
        assert not self.widget.client.is_ylog()

    def test_double_add_ignored(self):
        layer = self.add_layer_via_method()
        nobj = self.widget.ui.xAxisComboBox.count()
        layer = self.add_layer_via_method()
        self.assertEquals(self.widget.ui.xAxisComboBox.count(), nobj)

    def test_subsets_dont_duplicate_fields(self):
        layer = self.add_layer_via_method()
        nobj = self.widget.ui.xAxisComboBox.count()
        subset = layer.new_subset()
        subset.register()
        assert subset in self.widget.ui.layerTree
        self.assertEquals(self.widget.ui.xAxisComboBox.count(), nobj)

    def test_checkboxes_toggle_visbility(self):
        layer = self.add_layer_via_method()
        self.set_layer_checkbox(layer, Qt.Unchecked)
        assert not self.is_layer_visible(layer)
        self.set_layer_checkbox(layer, Qt.Checked)
        assert self.is_layer_visible(layer)


if __name__ == "__main__":
    unittest.main()