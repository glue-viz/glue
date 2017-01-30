from __future__ import absolute_import, division, print_function

import os

from qtpy import QtCore, QtWidgets
from glue.utils.qt import get_qapp
from glue.utils.qt import load_ui, connect_color
from glue.utils.qt.widget_properties import CurrentComboProperty, ValueProperty, connect_value, connect_current_combo
from glue.icons.qt import POINT_ICONS, symbol_icon


class ScatterLayerStyleWidget(QtWidgets.QWidget):

    size = ValueProperty('ui.value_size')
    symbol = CurrentComboProperty('ui.combo_symbol')
    alpha = ValueProperty('ui.slider_alpha', value_range=(0, 1))

    def __init__(self, layer_artist):

        super(ScatterLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._setup_symbol_combo()

        self.layer = layer_artist.layer

        # Set up connections
        self._connect_global()

        # Set initial values
        self.symbol = self.layer.style.marker
        self.size = self.layer.style.markersize
        self.ui.label_color.setColor(self.layer.style.color)
        self.alpha = self.layer.style.alpha

    def _connect_global(self):
        connect_current_combo(self.layer.style, 'marker', self.ui.combo_symbol)
        connect_value(self.layer.style, 'markersize', self.ui.value_size)
        connect_color(self.layer.style, 'color', self.ui.label_color)
        connect_value(self.layer.style, 'alpha', self.ui.slider_alpha, value_range=(0, 1))

    def _setup_symbol_combo(self):
        self._symbols = list(POINT_ICONS.keys())
        for idx, symbol in enumerate(self._symbols):
            icon = symbol_icon(symbol)
            self.ui.combo_symbol.addItem(icon, '', userData=symbol)
        self.ui.combo_symbol.setIconSize(QtCore.QSize(16, 16))

if __name__ == "__main__":
    app = get_qapp()
    options = ScatterLayerStyleWidget()
    options.show()
    app.exec_()
