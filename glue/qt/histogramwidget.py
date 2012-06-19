from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt, QVariant

import glue
import glue.message as msg
from glue.histogram_client import HistogramClient
from ui_histogramwidget import Ui_HistogramWidget

class HistogramWidget(QtGui.QMainWindow, glue.HubListener):

    def __init__(self, data, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.widget = QtGui.QWidget()
        self.setCentralWidget(self.widget)
        self.ui = Ui_HistogramWidget()
        self.ui.setupUi(self.widget)
        self.client = HistogramClient(data,
                                      self.ui.mplWidget.canvas.ax)

        self._connect()

    def _connect(self):
        ui = self.ui
        ui.attributeCombo.currentIndexChanged.connect(
            self._set_attribute_from_combo)
        ui.dataCombo.currentIndexChanged.connect(
            self._set_data_from_combo)
        ui.widthSpinBox.valueChanged.connect(
            self._update_bin_width)

    def _update_bin_width(self, width):
        self.client.set_width(width)

    def _update_attributes(self, data):
        comps = data.component_ids()
        combo = self.ui.attributeCombo
        try:
            combo.currentIndexChanged.disconnect()
        except TypeError:
            pass

        for comp in comps:
            combo.addItem(comp.label, userData = QVariant(comp))

        combo.currentIndexChanged.connect(self._set_attribute_from_combo)

    def _set_attribute_from_combo(self):
        combo = self.ui.attributeCombo
        index =  combo.currentIndex()
        attribute = combo.itemData(index).toPyObject()
        self.client.set_component(attribute)

    def _set_data_from_combo(self):
        self.client.set_data(self._current_data())

    def _current_data(self):
        combo = self.ui.dataCombo
        layer = combo.itemData(combo.currentIndex()).toPyObject()
        return layer

    def add_data(self, data):
        if self.data_present(data):
            return

        if self._first_data():
            self._update_attributes(data)

        combo = self.ui.dataCombo
        combo.addItem(data.label, userData = data)

        self.ui.layerTree.add_layer(data)

    def _first_data(self):
        return self.ui.dataCombo.count() == 0

    def data_present(self, data):
        for i in range(self.ui.dataCombo.count()):
            obj = self.ui.dataCombo.itemData(i).toPyObject()
            if data is obj:
                return True
        return False

    def register_to_hub(self, hub):
        self.client.register_to_hub(hub)
        self.ui.layerTree.register_to_hub(hub)

        hub.subscribe(self,
                      msg.DataCollectionAddMessage,
                      handler=lambda x:self.add_data(x.data),
                      filter = lambda x: x.sender is self.client.data)
