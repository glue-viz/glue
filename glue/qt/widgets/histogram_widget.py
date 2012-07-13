from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt, QVariant

from ...core.hub import HubListener
from ...core import message as msg
from ...clients.histogram_client import HistogramClient
from ..ui.histogramwidget import Ui_HistogramWidget
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import RectangleMode
from .data_viewer import DataViewer

class HistogramWidget(DataViewer):

    def __init__(self, data, parent=None):
        super(HistogramWidget, self).__init__(self, parent)

        self.central_widget = QtGui.QWidget()
        self.setCentralWidget(self.central_widget)
        self.ui = Ui_HistogramWidget()
        self.ui.setupUi(self.central_widget)
        self.make_toolbar()
        self.client = HistogramClient(data,
                                      self.ui.mplWidget.canvas.ax)
        self.ui.layerTree.data_collection = data
        self._connect()
        for d in data:
            self.add_data(d)

        self._tweak_geometry()

    def _tweak_geometry(self):
        self.central_widget.resize(400, 400)
        self.ui.splitter.setSizes([350, 120])
        self.resize(self.central_widget.size())

    def _connect(self):
        ui = self.ui
        cl = self.client
        ui.dataCombo.currentIndexChanged.connect(
            self._set_data_from_combo)
        ui.attributeCombo.currentIndexChanged.connect(
            self._set_attribute_from_combo)
        ui.binSpinBox.valueChanged.connect(
            self.client.set_nbins)
        ui.layerTree._layer_check_changed.connect(cl.set_layer_visible)
        ui.layerTree.linkButton.hide()
        ui.normalized_box.toggled.connect(cl.set_normalized)
        ui.autoscale_box.toggled.connect(cl.set_autoscale)

    def make_toolbar(self):
        result = GlueToolbar(self.ui.mplWidget.canvas, self,
                             name='Histogram')
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.ui.mplWidget.canvas.ax
        rect = RectangleMode(axes, release_callback=self._apply_roi)
        return [rect]

    def _apply_roi(self, mode):
        roi = mode.roi()
        self.client._apply_roi(roi)

    def _update_attributes(self, data=None):
        data = data or self._current_data()
        comps = data.components
        combo = self.ui.attributeCombo
        try:
            combo.currentIndexChanged.disconnect()
        except TypeError:
            pass

        combo.clear()
        for comp in comps:
            combo.addItem(comp.label, userData = QVariant(comp))

        combo.currentIndexChanged.connect(self._set_attribute_from_combo)
        combo.setCurrentIndex(0)
        self._set_attribute_from_combo()

    def _set_attribute_from_combo(self):
        combo = self.ui.attributeCombo
        index =  combo.currentIndex()
        attribute = combo.itemData(index).toPyObject()
        self.client.set_component(attribute)

    def _set_data_from_combo(self):
        self._update_attributes()
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
        super(HistogramWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)
        self.ui.layerTree.register_to_hub(hub)
        hub.subscribe(self,
                      msg.DataCollectionAddMessage,
                      handler=lambda x:self.add_data(x.data))

    def unregister(self, hub):
        for obj in [self.client, self.ui.layerTree, self]:
            hub.unsubscribe_all(obj)
