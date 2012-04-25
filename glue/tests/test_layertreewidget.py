import unittest
from threading import Thread

from PyQt4.QtGui import QApplication, QMainWindow
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

import glue
from glue.qt.layertreewidget import LayerTreeWidget

from glue.qt.qtutil import DebugClipboard


class SignalCatcher(object):
    def __init__(self):
        self.args = []
        self.kwargs = {}
        self.catches = 0

    def catch(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.catches += 1

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
        self.win.show()

    def tearDown(self):
        self.win.close()
        self.app.exit()

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

    def drag_layer_to_clipboard(self, layer, clipboard):
        tree = self.widget.layerTree
        QTest.mousePress(tree, Qt.LeftButton)
        QTest.mouseMove(clipboard)
        QTest.mouseRelease(clipboard, Qt.LeftButton)

    def layer_present(self, layer):
        """ Test that a layer exists in the widget """
        return self.widget.is_layer_present(layer)

    def layer_position(self, layer):
        widget_item = self.widget[layer]
        rect = self.widget.layerTree.visualItemRect(widget_item)
        return rect.center()

    def test_current_layer(self):
        layer = self.add_layer_via_method()
        item = self.widget[layer]
        self.widget.layerTree.setCurrentItem(item)
        self.assertIs(self.widget.current_layer(), layer)

    def test_current_layer_null(self):
        self.assertIs(None, self.widget.current_layer())

    def test_add_layer(self):
        """ Test that a layer exists once added """
        layer = self.add_layer_via_method()
        self.assertTrue(self.layer_present(layer))

    def test_remove_layer(self):
        """ Test that widget remove button works properly """
        layer = self.add_layer_via_method()
        self.remove_layer_via_button(layer)
        self.assertFalse(self.layer_present(layer))

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

    def test_orphan_subset_grabs_parent(self):
        """ Test that subset layers are added properly """
        layer = self.data[0]
        sub = glue.Subset(layer)
        self.widget.add_layer(sub)
        self.assertTrue(self.layer_present(sub))
        self.assertTrue(self.layer_present(layer))

    @unittest.expectedFailure
    def test_drag_functional(self):
        """still not implemented"""
        clipboard = DebugClipboard()
        clipboard.show()
        layer = self.add_layer_via_method()
        self.drag_layer_to_clipboard(layer, clipboard)
        self.assertTrue(clipboard.get_text() == layer.label)

    def test_signals(self):
        layer = self.add_layer_via_method()
        sc = SignalCatcher()
        self.widget.layer_check_changed.connect(sc.catch)
        item = self.widget[layer]
        item.setCheckState(0, Qt.Unchecked)
        self.assertEquals(sc.catches, 1)
        self.assertEquals(sc.args[0], layer)
        self.assertEquals(sc.args[1], False)

if __name__ == "__main__":
    unittest.main()