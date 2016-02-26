import os

import numpy as np

from glue.external.qt import QtGui

from glue.core import Subset
from glue.utils.qt import load_ui, update_combobox
from glue.utils.qt.widget_properties import (CurrentComboTextProperty,
                                             CurrentComboProperty,
                                             FloatLineProperty)
from glue.utils.qt.helpers import CUSTOM_QWIDGETS


__all__ = ['QLimitsWidget']


class QLimitsWidget(QtGui.QWidget):

    attribute = CurrentComboProperty('ui.combo_attribute')
    scale_mode = CurrentComboTextProperty('ui.combo_mode')
    percentile = CurrentComboProperty('ui.combo_mode')
    vlo = FloatLineProperty('ui.value_lower')
    vhi = FloatLineProperty('ui.value_upper')

    def __init__(self, parent=None):

        super(QLimitsWidget, self).__init__(parent=parent)

        self.ui = load_ui('attribute_limits.ui', self,
                          directory=os.path.dirname(__file__))

        self._setup_mode_combo()

        self.ui.combo_attribute.currentIndexChanged.connect(self._update_limits)
        self.ui.combo_mode.currentIndexChanged.connect(self._update_mode)

        self.ui.button_flip.clicked.connect(self._flip_limits)

        self.value_lower.editingFinished.connect(self._manual_edit)
        self.value_upper.editingFinished.connect(self._manual_edit)

        self._limits = {}
        self._callbacks = []


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._setup_attribute_combo()

    def _setup_attribute_combo(self):
        self.ui.combo_attribute.clear()
        if isinstance(self.data, Subset):
            components = self.data.data.visible_components
        else:
            components = self.data.visible_components
        label_data = [(comp.label, comp) for comp in components]
        update_combobox(self.ui.combo_attribute, label_data)

    def _setup_mode_combo(self):
        self.ui.combo_mode.clear()
        self.ui.combo_mode.addItem("Min/Max", userData=100)
        self.ui.combo_mode.addItem("99.5%", userData=99.5)
        self.ui.combo_mode.addItem("99%", userData=99)
        self.ui.combo_mode.addItem("95%", userData=95)
        self.ui.combo_mode.addItem("90%", userData=90)
        self.ui.combo_mode.addItem("Custom", userData=None)
        self.ui.combo_mode.setCurrentIndex(-1)

    def _flip_limits(self):
        # We only need to emit a signal for one of the limits
        self.ui.value_lower.blockSignals(True)
        self.vlo, self.vhi = self.vhi, self.vlo
        self.ui.value_lower.blockSignals(False)
        self.notify_callbacks()

    def _manual_edit(self):
        self._cache_limits()
        self.notify_callbacks()

    def _update_mode(self):
        if self.scale_mode == 'Custom':
            self.ui.value_lower.setEnabled(True)
            self.ui.value_upper.setEnabled(True)
        else:
            self.ui.value_lower.setEnabled(False)
            self.ui.value_upper.setEnabled(False)
            self._auto_limits()
            self._cache_limits()
            self.notify_callbacks()

    def _cache_limits(self):
        self._limits[self.attribute] = self.scale_mode, self.vlo, self.vhi

    def _update_limits(self):
        # We only need to emit a signal for one of the limits
        self.ui.combo_mode.blockSignals(True)
        self.ui.value_lower.blockSignals(True)
        self.ui.value_upper.blockSignals(True)
        if self.attribute in self._limits:
            self.ui.value_lower.blockSignals(True)
            self.scale_mode, self.vlo, self.vhi = self._limits[self.attribute]
            self.ui.value_lower.blockSignals(False)
        else:
            self.scale_mode = 'Min/Max'
            self._update_mode()
        self.ui.combo_mode.blockSignals(False)
        self.ui.value_lower.blockSignals(False)
        self.ui.value_upper.blockSignals(False)

    def _auto_limits(self):
        # We only need to emit a signal for one of the limits
        self.ui.value_lower.blockSignals(True)
        self.ui.value_upper.blockSignals(True)
        exclude = (100 - self.percentile) / 2.
        self.vlo = np.nanpercentile(self.data[self.attribute], exclude)
        self.vhi = np.nanpercentile(self.data[self.attribute], 100 - exclude)
        self.ui.value_lower.blockSignals(False)
        self.ui.value_upper.blockSignals(False)
        print(self.vlo, self.vhi)

    def connect(self, function):
        self._callbacks.append(function)

    def notify_callbacks(self):
        print("NOTIFY")
        for func in self._callbacks:
            func()

CUSTOM_QWIDGETS.append('QLimitsWidget')

if __name__ == "__main__":

    from glue.external.qt import get_qapp
    from glue.core import Data

    def cb():
        print("CALLBACK!")

    data1 = np.random.random(1000)
    data2 = np.random.random(1000)
    data = Data(x=data1, y=data2)
    app = get_qapp()
    widget = QLimitsWidget(data=data)
    widget.connect(cb)
    widget.show()
    app.exec_()
