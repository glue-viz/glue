import sys

from mock import MagicMock
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest

from ..histogramwidget import HistogramWidget
from ....clients.histogram_client import HistogramClient
from .... import core
from ....tests import example_data

class TestHistogramWidget(object):
    def setup_method(self, method):
        self.data = self.mock_data()
        self.collect = core.data_collection.DataCollection([self.data])
        self.widget = HistogramWidget(self.collect)
        self.widget.client = MagicMock(spec=HistogramClient)
        self.widget.client.data = self.collect

    def mock_data(self):
        data = example_data.test_histogram_data()
        return data

    def teardown_method(self, method):
        del self.widget

    def setup_class(self):
        self.app = QApplication(sys.argv)

    def teardown_class(self):
        del self.app

    def set_dummy_attributes(self):
        combo = self.widget.ui.attributeCombo
        combo.addItem("Dummy1", userData = "Dum1")
        combo.addItem("Dummy2", userData = "Dum2")

    def set_dummy_data(self):
        combo = self.widget.ui.dataCombo
        d1 = MagicMock()
        d2 = MagicMock()
        combo.addItem("DataDummy1", userData = d1)
        combo.addItem("DataDummy2", userData = d2)

    def set_up_hub(self):
        hub = core.hub.Hub()
        self.collect.register_to_hub(hub)
        self.widget.register_to_hub(hub)
        return hub

    def test_update_attributes(self):

        self.widget._update_attributes(self.data)
        combo = self.widget.ui.attributeCombo
        comps = self.data.components
        assert combo.count() == len(comps)
        for i in range(combo.count()):
            assert combo.itemData(i).toPyObject() in comps

    def test_attribute_set_with_combo(self):
        self.set_dummy_attributes()
        self.widget.ui.attributeCombo.setCurrentIndex(1)
        obj = self.widget.ui.attributeCombo.itemData(1).toPyObject()
        self.widget.client.set_component.assert_called_with(obj)

        obj = self.widget.ui.attributeCombo.itemData(0).toPyObject()
        self.widget.ui.attributeCombo.setCurrentIndex(0)
        self.widget.client.set_component.assert_called_with(obj)

    def test_data_set_with_combo(self):
        self.set_dummy_data()
        self.widget.ui.dataCombo.setCurrentIndex(1)
        obj = self.widget.ui.dataCombo.itemData(1).toPyObject()
        self.widget.client.set_data.assert_called_with(obj)

        self.widget.ui.dataCombo.setCurrentIndex(0)
        obj = self.widget.ui.dataCombo.itemData(0).toPyObject()
        self.widget.client.set_data.assert_called_with(obj)

    def test_add_data_populates_combo(self):
        self.widget.add_data(self.data)
        assert self.widget.ui.dataCombo.itemData(0).toPyObject() is \
            self.data

    def test_data_message_received(self):
        hub = self.set_up_hub()
        d2 = self.mock_data()
        self.collect.append(d2)
        assert self.widget.data_present(d2)

    def test_data_message_populates_layertree(self):
        hub = self.set_up_hub()
        d2 = self.mock_data()
        self.collect.append(d2)
        assert d2 in self.widget.ui.layerTree

    def test_attributes_populated_after_first_data_add(self):
        d2 = self.data
        self.collect.append(d2)
        self.widget.add_data(d2)

        comps = d2.components
        assert self.widget.ui.attributeCombo.count() == len(comps)
        for i in range(self.widget.ui.attributeCombo.count()):
            data = self.widget.ui.attributeCombo.itemData(i).toPyObject()
            assert data in comps

    def test_double_add_ignored(self):
        self.widget.add_data(self.data)
        ct = self.widget.ui.dataCombo.count()
        self.widget.add_data(self.data)
        assert ct == self.widget.ui.dataCombo.count()
