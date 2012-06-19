import unittest
import sys

from mock import MagicMock
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest

import glue
from glue.qt.histogramwidget import HistogramWidget
from glue.histogram_client import HistogramClient

@unittest.skip("not implemented")
class TestHistogramWidget(unittest.TestCase):
    def setUp(self):
        self.app = QApplication(sys.argv)
        self.data = self.mock_data()
        self.collect = glue.DataCollection([self.data])
        self.widget = HistogramWidget(self.collect)
        self.widget.client = MagicMock(spec=HistogramClient)
        self.widget.client.data = self.collect

    def mock_data(self):
        data = glue.example_data.test_histogram_data()
        return data

        data = MagicMock(spec=glue.Data)
        comp1 = MagicMock(spec=glue.Component)
        comp2 = MagicMock(spec=glue.Component)
        comp1.label = 'dummy1'
        comp2.label = 'dummy2'
        data.component_ids.return_value = [comp1, comp2]
        data.label = "Mock data"
        return data

    def tearDown(self):
        del self.widget
        del self.app

    def set_dummy_attributes(self):
        combo = self.widget.ui.attributeCombo
        combo.addItem("Dummy1", userData = "Dum1")
        combo.addItem("Dummy2", userData = "Dum2")

    def set_dummy_data(self):
        combo = self.widget.ui.dataCombo
        combo.addItem("DataDummy1", userData = "DataDum1")
        combo.addItem("DataDummy2", userData = "DataDum2")

    def set_up_hub(self):
        hub = glue.Hub()
        self.collect.register_to_hub(hub)
        self.widget.register_to_hub(hub)
        return hub

    def test_update_attributes(self):

        self.widget._update_attributes(self.data)
        combo = self.widget.ui.attributeCombo
        comps = self.data.component_ids()
        self.assertTrue(combo.count() == len(comps))
        for i in range(combo.count()):
            self.assertTrue(combo.itemData(i).toPyObject() in comps)

    def test_attribute_set_with_combo(self):
        self.set_dummy_attributes()
        self.widget.ui.attributeCombo.setCurrentIndex(0)
        self.widget.client.set_component.assert_called_with('Dum1')
        self.widget.ui.attributeCombo.setCurrentIndex(1)
        self.widget.client.set_component.assert_called_with('Dum2')

    def test_data_set_with_combo(self):
        self.set_dummy_data()
        self.widget.ui.dataCombo.setCurrentIndex(0)
        self.widget.client.set_data.assert_called_with('DataDum1')
        self.widget.ui.dataCombo.setCurrentIndex(1)
        self.widget.client.set_data.assert_called_with('DataDum2')

    def test_width_spinner_calls_client(self):
        self.widget.ui.widthSpinBox.setValue(50)
        self.widget.client.set_width.assert_called_with(50)

    def test_add_data_populates_combo(self):
        self.widget.add_data(self.data)
        self.assertIs(self.widget.ui.dataCombo.itemData(0).toPyObject(),
                      self.data)

    def test_data_message_received(self):
        hub = self.set_up_hub()
        d2 = self.mock_data()
        self.collect.append(d2)
        self.assertTrue(self.widget.data_present(d2))

    def test_data_message_populates_layertree(self):
        hub = self.set_up_hub()
        d2 = self.mock_data()
        self.collect.append(d2)
        self.assertTrue(d2 in self.widget.ui.layerTree)

    def test_attributes_populated_after_first_data_add(self):
        d2 = self.mock_data()
        self.collect.append(d2)
        self.widget.add_data(d2)

        comps = d2.component_ids()
        self.assertEquals(self.widget.ui.attributeCombo.count(), len(comps))
        for i in range(self.widget.ui.attributeCombo.count()):
            data = self.widget.ui.attributeCombo.itemData(i).toPyObject()
            self.assertTrue(data in comps)

    def test_double_add_ignored(self):
        self.assertFalse(self.widget.data_present(self.data))
        self.widget.add_data(self.data)
        self.assertTrue(self.widget.data_present(self.data))
        self.widget.add_data(self.data)

if __name__ == "__main__":
    unittest.main()