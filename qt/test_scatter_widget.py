import unittest

from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

import cloudviz as cv
from scatterwidget import ScatterWidget



class TestScatterWidget(unittest.TestCase):

    def setUp(self):
        import sys
        self.app = QApplication(sys.argv)
        self.hub = cv.Hub()
        self.data = cv.example_data.pipe()[:2]
        self.collect = cv.DataCollection()
        self.widget = ScatterWidget(self.collect)
        self.win = QMainWindow()
        self.win.setCentralWidget(self.widget)
        self.connect_to_hub()
        self.win.show()

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

    def is_layer_present(self, layer):
        if not self.widget.ui.layerTree.is_layer_present(layer):
            return False
        return self.widget.client.is_layer_present(layer)

    def test_hub_data_add_is_ignored(self):
        layer = self.add_layer_via_hub()
        self.assertFalse(self.widget.ui.layerTree.is_layer_present(layer))
        self.assertFalse(self.widget.client.is_layer_present(layer))

    def test_valid_add_data_via_method(self):
        layer = self.add_layer_via_method()
        self.assertTrue(self.is_layer_present(layer))

    def test_add_first_data_updates_combos(self):
        layer = self.add_layer_via_method()
        xatt = str(self.widget.ui.xAxisComboBox.currentText())
        yatt = str(self.widget.ui.yAxisComboBox.currentText())
        self.assertEquals(xatt, 'A_Vb')
        self.assertEquals(yatt, 'A_Vb')

    def test_flip_x(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.xFlipCheckBox, Qt.LeftButton)
        self.assertTrue(self.widget.client.is_xflip())
        QTest.mouseClick(self.widget.ui.xFlipCheckBox, Qt.LeftButton)
        self.assertFalse(self.widget.client.is_xflip())

    def test_flip_y(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.yFlipCheckBox, Qt.LeftButton)
        self.assertTrue(self.widget.client.is_yflip())
        QTest.mouseClick(self.widget.ui.yFlipCheckBox, Qt.LeftButton)
        self.assertFalse(self.widget.client.is_yflip())

    def test_log_x(self):
        layer = self.add_layer_via_method()
        QTest.mouseClick(self.widget.ui.xLogCheckBox, Qt.LeftButton)
        self.assertTrue(self.widget.client.is_xlog())
        QTest.mouseClick(self.widget.ui.xLogCheckBox, Qt.LeftButton)
        self.assertFalse(self.widget.client.is_xlog())

    def test_log_y(self):
        QTest.mouseClick(self.widget.ui.yLogCheckBox, Qt.LeftButton)
        self.assertTrue(self.widget.client.is_ylog())
        QTest.mouseClick(self.widget.ui.yLogCheckBox, Qt.LeftButton)
        self.assertFalse(self.widget.client.is_ylog())


if __name__ == "__main__":
    unittest.main()