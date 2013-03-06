from functools import partial

from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from ... import core

from ...clients.scatter_client import ScatterClient
from ..glue_toolbar import GlueToolbar
from ..mouse_mode import (RectangleMode, CircleMode,
                          PolyMode, HRangeMode, VRangeMode)
from ...core.callback_property import add_callback

from ..ui.scatterwidget import Ui_ScatterWidget
from .data_viewer import DataViewer
from .mpl_widget import MplWidget
from ..widget_properties import ButtonProperty, FloatLineProperty
from ..qtutil import pretty_number


WARN_SLOW = 1000000  # max number of points which render quickly


def connect_bool_button(client, prop, widget):
    add_callback(client, prop, widget.setChecked)
    widget.toggled.connect(partial(setattr, client, prop))


def connect_float_edit(client, prop, widget):
    v = QtGui.QDoubleValidator(None)
    v.setDecimals(4)
    widget.setValidator(v)

    def update_prop():
        val = widget.text()
        try:
            setattr(client, prop, float(val))
        except ValueError:
            setattr(client, prop, 0)

    def update_widget(val):
        widget.setText(pretty_number(val))

    add_callback(client, prop, update_widget)
    widget.editingFinished.connect(update_prop)
    update_widget(getattr(client, prop))


class ScatterWidget(DataViewer):
    LABEL = "Scatter Plot"

    xlog = ButtonProperty('ui.xLogCheckBox')
    ylog = ButtonProperty('ui.yLogCheckBox')
    xflip = ButtonProperty('ui.xFlipCheckBox')
    yflip = ButtonProperty('ui.yFlipCheckBox')
    xmin = FloatLineProperty('ui.xmin')
    xmax = FloatLineProperty('ui.xmax')
    ymin = FloatLineProperty('ui.ymin')
    ymax = FloatLineProperty('ui.ymax')
    hidden = ButtonProperty('ui.hidden_attributes')

    def __init__(self, data, parent=None):
        super(ScatterWidget, self).__init__(data, parent)
        self.central_widget = MplWidget()
        self.option_widget = QtGui.QWidget()

        self.setCentralWidget(self.central_widget)

        self.ui = Ui_ScatterWidget()
        self.ui.setupUi(self.option_widget)
        self._tweak_geometry()
        self._collection = data

        self.client = ScatterClient(self._collection,
                                    self.central_widget.canvas.fig,
                                    artist_container=self._container)
        self._connect()
        self.unique_fields = set()
        self.make_toolbar()
        self.statusBar().setSizeGripEnabled(False)
        self.setFocusPolicy(Qt.StrongFocus)

    def _tweak_geometry(self):
        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

    def _connect(self):
        ui = self.ui
        cl = self.client

        connect_bool_button(cl, 'xlog', ui.xLogCheckBox)
        connect_bool_button(cl, 'ylog', ui.yLogCheckBox)
        connect_bool_button(cl, 'xflip', ui.xFlipCheckBox)
        connect_bool_button(cl, 'yflip', ui.yFlipCheckBox)

        ui.xAxisComboBox.currentIndexChanged.connect(self.update_xatt)
        ui.yAxisComboBox.currentIndexChanged.connect(self.update_yatt)
        ui.hidden_attributes.toggled.connect(lambda x: self._update_combos())
        ui.swapAxes.clicked.connect(self.swap_axes)
        ui.snapLimits.clicked.connect(cl.snap)

        connect_float_edit(cl, 'xmin', ui.xmin)
        connect_float_edit(cl, 'xmax', ui.xmax)
        connect_float_edit(cl, 'ymin', ui.ymin)
        connect_float_edit(cl, 'ymax', ui.ymax)

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
        result = GlueToolbar(self.central_widget.canvas, self,
                             name='Scatter Plot')
        for mode in self._mouse_modes():
            result.add_mode(mode)
        self.addToolBar(result)
        return result

    def _mouse_modes(self):
        axes = self.client.axes
        rect = RectangleMode(axes, roi_callback=self.apply_roi)
        xra = HRangeMode(axes, roi_callback=self.apply_roi)
        yra = VRangeMode(axes, roi_callback=self.apply_roi)
        circ = CircleMode(axes, roi_callback=self.apply_roi)
        poly = PolyMode(axes, roi_callback=self.apply_roi)
        return [rect, xra, yra, circ, poly]

    def apply_roi(self, mode):
        roi = mode.roi()
        self.client.apply_roi(roi)

    def _update_combos(self):
        """ Update combo boxes to current attribute fields"""
        layer_ids = []
        for l in self.client.data:
            if not self.client.is_layer_present(l):
                continue
            for lid in self.client.plottable_attributes(
                    l, show_hidden=self.hidden):
                if lid not in layer_ids:
                    layer_ids.append(lid)

        xcombo = self.ui.xAxisComboBox
        ycombo = self.ui.yAxisComboBox

        xcombo.blockSignals(True)
        ycombo.blockSignals(True)

        xid = xcombo.itemData(xcombo.currentIndex())
        yid = ycombo.itemData(ycombo.currentIndex())

        xcombo.clear()
        ycombo.clear()

        for lid in layer_ids:
            xcombo.addItem(lid.label, userData=lid)
            ycombo.addItem(lid.label, userData=lid)

        for index, combo in zip([xid, yid], [xcombo, ycombo]):
            try:
                combo.setCurrentIndex(layer_ids.index(index))
            except ValueError:
                combo.setCurrentIndex(0)

        xcombo.blockSignals(False)
        ycombo.blockSignals(False)

    def add_data(self, data):
        """Add a new data set to the widget

        :rtype: bool
        Returns True if the addition was expected, False otherwise
        """
        if self.client.is_layer_present(data):
            return

        if data.size > WARN_SLOW and not self._confirm_large_data(data):
            return False

        first_layer = self.client.layer_count == 0

        self.client.add_data(data)
        self._update_combos()

        if first_layer:  # forces both x and y axes to be rescaled
            self.update_xatt(None)
            self.update_yatt(None)

            self.ui.xAxisComboBox.setCurrentIndex(0)
            if len(data.visible_components) > 1:
                self.ui.yAxisComboBox.setCurrentIndex(1)
            else:
                self.ui.yAxisComboBox.setCurrentIndex(0)

        self._update_window_title()
        return True

    def add_subset(self, subset):
        """Add a subset to the widget

        :rtype: bool:
        Returns True if the addition was accepted, False otherwise
        """
        if self.client.is_layer_present(subset):
            return

        data = subset.data
        if data.size > WARN_SLOW and not self._confirm_large_data(data):
            return False

        first_layer = self.client.layer_count == 0

        self.client.add_layer(subset)
        self._update_combos()

        if first_layer:  # forces both x and y axes to be rescaled
            self.update_xatt(None)
            self.update_yatt(None)

        self.ui.xAxisComboBox.setCurrentIndex(0)
        if len(data.visible_components) > 1:
            self.ui.yAxisComboBox.setCurrentIndex(1)
        else:
            self.ui.yAxisComboBox.setCurrentIndex(0)

        self._update_window_title()
        return True

    def register_to_hub(self, hub):
        super(ScatterWidget, self).register_to_hub(hub)
        self.client.register_to_hub(hub)
        hub.subscribe(self, core.message.DataUpdateMessage,
                      lambda x: self._sync_labels())
        hub.subscribe(self, core.message.ComponentsChangedMessage,
                      lambda x: self._update_combos())

    def unregister(self, hub):
        hub.unsubscribe_all(self.client)
        hub.unsubscribe_all(self)

    def swap_axes(self):
        xid = self.ui.xAxisComboBox.currentIndex()
        yid = self.ui.yAxisComboBox.currentIndex()
        xlog = self.ui.xLogCheckBox.isChecked()
        ylog = self.ui.yLogCheckBox.isChecked()
        xflip = self.ui.xFlipCheckBox.isChecked()
        yflip = self.ui.yFlipCheckBox.isChecked()

        self.ui.xAxisComboBox.setCurrentIndex(yid)
        self.ui.yAxisComboBox.setCurrentIndex(xid)
        self.ui.xLogCheckBox.setChecked(ylog)
        self.ui.yLogCheckBox.setChecked(xlog)
        self.ui.xFlipCheckBox.setChecked(yflip)
        self.ui.yFlipCheckBox.setChecked(xflip)

    def update_xatt(self, index):
        combo = self.ui.xAxisComboBox
        component_id = combo.itemData(combo.currentIndex())
        assert isinstance(component_id, core.data.ComponentID)
        self.client.xatt = component_id

    def update_yatt(self, index):
        combo = self.ui.yAxisComboBox
        component_id = combo.itemData(combo.currentIndex())
        assert isinstance(component_id, core.data.ComponentID)
        self.client.yatt = component_id

    def _update_window_title(self):
        data = self.client.data
        label = ', '.join([d.label for d in data if
                           self.client.is_visible(d)])
        self.setWindowTitle(label)

    def _sync_labels(self):
        self._update_window_title()

    def __str__(self):
        return "Scatter Widget"

    def options_widget(self):
        return self.option_widget
