from PyQt4.QtCore import Qt
from PyQt4 import QtGui

import glue
import glue.message as msg
from glue.scatter_client import ScatterClient

from ui_scatterwidget import Ui_ScatterWidget
from linker_dialog import LinkerDialog

class ScatterWidget(QtGui.QWidget, glue.HubListener) :
    def __init__(self, data, parent=None):
        QtGui.QWidget.__init__(self, parent)
        glue.HubListener.__init__(self)
        self.ui = Ui_ScatterWidget()
        self.ui.setupUi(self)
        self.client = ScatterClient(data,
                                    self.ui.mplWidget.canvas.fig,
                                    self.ui.mplWidget.canvas.ax)
        self.connect()
        self.unique_fields = set()

    def connect(self):
        ui = self.ui
        cl = self.client

        ui.xLogCheckBox.stateChanged.connect(
            lambda x: cl.set_xlog(x==Qt.Checked))
        ui.yLogCheckBox.stateChanged.connect(
            lambda x: cl.set_ylog(x==Qt.Checked))
        ui.xFlipCheckBox.stateChanged.connect(
            lambda x: cl.set_xflip(x==Qt.Checked))
        ui.yFlipCheckBox.stateChanged.connect(
            lambda x: cl.set_yflip(x==Qt.Checked))
        ui.xAxisComboBox.currentIndexChanged.connect(self.update_xatt)
        ui.yAxisComboBox.currentIndexChanged.connect(self.update_yatt)
        ui.layerTree.layer_check_changed.connect(cl.set_visible)

    def update_combos(self, layer):
        """ Update combo boxes to incorporate attribute fields in layer"""
        fields = self.client.plottable_attributes(layer)
        xcombo = self.ui.xAxisComboBox
        ycombo = self.ui.yAxisComboBox

        for f in fields:
            key = (layer.data, f)
            if key not in self.unique_fields:
                xcombo.addItem(f)
                ycombo.addItem(f)
            self.unique_fields.add(key)

    def add_layer(self, layer):
        if layer in self.ui.layerTree:
            return

        self.ui.layerTree.add_layer(layer)
        self.client.add_layer(layer)
        self.update_combos(layer)

        for sub in layer.data.subsets:
            self.ui.layerTree.add_layer(sub)

    def register_to_hub(self, hub):
        self.client.register_to_hub(hub)
        self.ui.layerTree.register_to_hub(hub)

    def update_xatt(self, index):
        combo = self.ui.xAxisComboBox
        att = str(combo.currentText())
        self.client.set_xdata(att)

    def update_yatt(self, index):
        combo = self.ui.yAxisComboBox
        att = str(combo.currentText())
        self.client.set_ydata(att)
