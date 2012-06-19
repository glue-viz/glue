from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import glue
from glue.scatter_client import ScatterClient
from glue.qt.glue_toolbar import GlueToolbar
from glue.qt.mouse_mode import RectangleMode, CircleMode, PolyMode

from ui_scatterwidget import Ui_ScatterWidget


class ScatterWidget(QtGui.QMainWindow, glue.HubListener):
    def __init__(self, data, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        glue.HubListener.__init__(self)
        self.central_widget = QtGui.QWidget()
        self.setCentralWidget(self.central_widget)
        self.ui = Ui_ScatterWidget()
        self.ui.setupUi(self.central_widget)
        self._tweak_geometry()
        self.client = ScatterClient(data,
                                    self.ui.mplWidget.canvas.fig,
                                    self.ui.mplWidget.canvas.ax)
        self._data = data
        self._connect()
        self.unique_fields = set()
        self.make_toolbar()
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def _tweak_geometry(self):
        self.central_widget.resize(400,400)
        self.ui.splitter.setSizes([320, 150])
        self.resize(self.central_widget.size())

    def _connect(self):
        ui = self.ui
        cl = self.client

        ui.xLogCheckBox.stateChanged.connect(
            lambda x: cl.set_xlog(x == Qt.Checked))
        ui.yLogCheckBox.stateChanged.connect(
            lambda x: cl.set_ylog(x == Qt.Checked))
        ui.xFlipCheckBox.stateChanged.connect(
            lambda x: cl.set_xflip(x == Qt.Checked))
        ui.yFlipCheckBox.stateChanged.connect(
            lambda x: cl.set_yflip(x == Qt.Checked))
        ui.xAxisComboBox.currentIndexChanged.connect(self.update_xatt)
        ui.yAxisComboBox.currentIndexChanged.connect(self.update_yatt)
        ui.layerTree._layer_check_changed.connect(cl.set_visible)
        ui.layerTree.layerAddButton.pressed.disconnect()
        ui.layerTree.layerAddButton.released.connect(self._add_data)
        ui.layerTree.linkButton.hide()

    def _add_data(self):
        choices = dict([(d.label, d) for d in self._data])
        dialog = QtGui.QInputDialog()
        choice, isok = dialog.getItem(self, "Data Chooser | Scatter Plot",
                                      "Choose a data set to add",
                                      choices.keys())
        if not isok:
            return
        data = choices[str(choice)]
        self.add_layer(data)

    def make_toolbar(self):
        result = GlueToolbar(self.ui.mplWidget.canvas, self,
                             name='Scatter Plot')
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.ui.mplWidget.canvas.ax
        rect = RectangleMode(axes, release_callback=self._apply_roi)
        circ = CircleMode(axes, release_callback=self._apply_roi)
        poly = PolyMode(axes, release_callback=self._apply_roi)
        return [rect, circ, poly]

    def _apply_roi(self, mode):
        roi = mode.roi()
        self.client._apply_roi(roi)

    def update_combos(self, layer):
        """ Update combo boxes to incorporate attribute fields in layer"""
        layer_ids = self.client.plottable_attributes(layer)
        xcombo = self.ui.xAxisComboBox
        ycombo = self.ui.yAxisComboBox

        for lid in layer_ids:
            if lid not in self.unique_fields:
                xcombo.addItem(lid.label, userData=lid)
                ycombo.addItem(lid.label, userData=lid)
            self.unique_fields.add(lid)

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
        component_id = combo.itemData(combo.currentIndex()).toPyObject()
        assert isinstance(component_id, glue.data.ComponentID)
        self.client.set_xdata(component_id)

    def update_yatt(self, index):
        combo = self.ui.yAxisComboBox
        component_id = combo.itemData(combo.currentIndex()).toPyObject()
        assert isinstance(component_id, glue.data.ComponentID)
        self.client.set_ydata(component_id)

    def __str__(self):
        return "Scatter Widget"
