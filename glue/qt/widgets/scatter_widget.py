from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from ... import core

from ...clients.scatter_client import ScatterClient
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import RectangleMode, CircleMode, PolyMode

from ..ui.scatterwidget import Ui_ScatterWidget
from .data_viewer import DataViewer


WARN_SLOW = 250000  # max number of points which render quickly


class ScatterWidget(DataViewer):
    LABEL = "Scatter Plot"

    def __init__(self, data, parent=None):
        super(ScatterWidget, self).__init__(data, parent)
        self.central_widget = QtGui.QWidget()
        self.setCentralWidget(self.central_widget)
        self.ui = Ui_ScatterWidget()
        self.ui.setupUi(self.central_widget)
        self._tweak_geometry()
        #set up a clean data collection for scatter client
        #lets us screen incoming data objects for size
        self._clean_collection = core.DataCollection()
        self._collection = data
        self.client = ScatterClient(self._clean_collection,
                                    self.ui.mplWidget.canvas.fig,
                                    self.ui.mplWidget.canvas.ax)
        self._connect()
        self.unique_fields = set()
        self.make_toolbar()
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.ui.layerTree.set_checkable(True)

    def _tweak_geometry(self):
        self.central_widget.resize(400, 400)
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
        ui.layerTree.layerAddButton.released.connect(self._choose_add_data)
        ui.layerTree.linkButton.hide()

    def _choose_add_data(self):
        choices = dict([(d.label, d) for d in self._collection])
        dialog = QtGui.QInputDialog()
        choice, isok = dialog.getItem(self, "Data Chooser | Scatter Plot",
                                      "Choose a data set to add",
                                      choices.keys())
        if not isok:
            return
        data = choices[str(choice)]
        self.add_data(data)

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

    def add_data(self, data):
        if data in self._clean_collection:
            return

        if data.size > WARN_SLOW and \
           not self._confirm_large_data(data):
            return

        first_layer = len(self._clean_collection) == 0
        self._clean_collection.append(data)
        self.update_combos(data)

        if first_layer:  # forces both x and y axes to be rescaled
            self.update_xatt(None)
            self.update_yatt(None)

    def register_to_hub(self, hub):
        super(ScatterWidget, self).register_to_hub(hub)
        self.ui.layerTree.setup(self._clean_collection, hub)
        self._clean_collection.register_to_hub(hub)
        self.client.register_to_hub(hub)

    def unregister(self, hub):
        self.ui.layerTree.unregister(hub)
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self._clean_collection)

    def update_xatt(self, index):
        combo = self.ui.xAxisComboBox
        component_id = combo.itemData(combo.currentIndex())
        assert isinstance(component_id, core.data.ComponentID)
        self.client.set_xdata(component_id)

    def update_yatt(self, index):
        combo = self.ui.yAxisComboBox
        component_id = combo.itemData(combo.currentIndex())
        assert isinstance(component_id, core.data.ComponentID)
        self.client.set_ydata(component_id)

    def _confirm_large_data(self, data):
        warn_msg = ("WARNING: Data set has %i points, and may render slowly."
                    " Continue?" % data.size)
        title = "Add large data set?"
        ok = QtGui.QMessageBox.Ok
        cancel = QtGui.QMessageBox.Cancel
        buttons = ok | cancel
        result = QtGui.QMessageBox.question(self, title, warn_msg,
                                            buttons=buttons,
                                            defaultButton=cancel)
        return result == ok

    def __str__(self):
        return "Scatter Widget"
