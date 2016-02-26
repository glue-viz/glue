from glue.external.qt import QtGui

from glue.utils.qt import load_ui
from glue.utils.qt.widget_properties import (CurrentComboTextProperty,
                                             CurrentComboProperty,
                                             FloatLineProperty)
from glue.utils.qt.helpers import CUSTOM_QWIDGETS


__all__ = ['QLimitsWidget']


class QLimitsWidget(QtGui.QWidget):

    mode = CurrentComboTextProperty('ui.combo_mode')
    percentile = CurrentComboProperty('ui.combo_mode')
    vlo = FloatLineProperty('ui.value_lower')
    vhi = FloatLineProperty('ui.value_upper')

    def __init__(self, data=None, parent=None):

        super(QLimitsWidget, self).__init__(parent=parent)

        self.ui = load_ui('scale_widget.ui', self)

        self._setup_mode_combo()

        self.ui.combo_mode.currentIndexChanged.connect(self._update_mode)
        self.ui.button_flip.clicked.connect(self._flip_limits)

        self.data = data

        self._callbacks = []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._auto_limits()

    def _setup_mode_combo(self):
        self.ui.combo_mode.clear()
        self.ui.combo_mode.addItem("Min/Max", userData=100)
        self.ui.combo_mode.addItem("99.5%", userData=99.5)
        self.ui.combo_mode.addItem("99%", userData=99)
        self.ui.combo_mode.addItem("95%", userData=95)
        self.ui.combo_mode.addItem("90%", userData=90)
        self.ui.combo_mode.addItem("Custom", userData=None)

    def _flip_limits(self):
        # We only need to emit a signal for one of the limits
        self.ui.value_lower.blockSignals(True)
        self.vlo, self.vhi = self.vhi, self.vlo
        self.ui.value_lower.blockSignals(False)

    def _update_mode(self):
        if self.mode == 'Custom':
            self.ui.value_lower.setEnabled(True)
            self.ui.value_upper.setEnabled(True)
        else:
            self.ui.value_lower.setEnabled(False)
            self.ui.value_upper.setEnabled(False)
            self._auto_limits()

    def _auto_limits(self):
        # We only need to emit a signal for one of the limits
        self.ui.value_lower.blockSignals(True)
        exclude = (100 - self.percentile) / 2.
        self.vlo = np.percentile(self.data, exclude)
        self.vhi = np.percentile(self.data, 100 - exclude)
        self.ui.value_lower.blockSignals(False)

    def connect(self, function):
        self.value_lower.editingFinished.connect(function)
        self.value_upper.editingFinished.connect(function)
        self.button_flip.editingFinished.connect(function)

CUSTOM_QWIDGETS.append('QLimitsWidget')

if __name__ == "__main__":

    import numpy as np
    from glue.external.qt import get_qapp

    data = np.random.random(1000)

    app = get_qapp()
    widget = QLimitsWidget(data)
    widget.show()
    app.exec_()
