from PyQt4.QtCore import Qt
from PyQt4 import QtGui

import cloudviz as cv
import cloudviz.message as msg
from cloudviz.scatter_client import ScatterClient

from ui_scatterwidget import Ui_ScatterWidget
from linker_dialog import LinkerDialog

class ScatterWidget(QtGui.QWidget, cv.HubListener) :
    def __init__(self, data, parent=None):
        QtGui.QWidget.__init__(self, parent)
        cv.HubListener.__init__(self)
        self.ui = Ui_ScatterWidget()
        self.ui.setupUi(self)
        self.client = ScatterClient(data,
                                    self.ui.mplWidget.canvas.fig,
                                    self.ui.mplWidget.canvas.ax)
        self.layer_dict = {}
        self.connect()

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

    def update_combos(self, layer):
        """ Update combo boxes to incorporate attribute fields in layer"""
        fields = self.client.plottable_attributes(layer)
        xcombo = self.ui.xAxisComboBox
        ycombo = self.ui.yAxisComboBox

        for f in fields:
            key = (layer, f)
            if xcombo.findData(key) == -1:
                xcombo.addItem(f, userData = key)
            if ycombo.findData(key) == -1:
                ycombo.addItem(f, userData = key)

    def add_layer(self, layer):
        self.ui.layerTree.add_layer(layer)
        self.client.init_layer(layer)
        self.update_combos(layer)

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
