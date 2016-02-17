from __future__ import absolute_import, division, print_function

import os
from time import ctime

from glue.external.qt import QtCore, QtGui, get_qapp
from glue import core
from glue import config
from glue.utils.qt import load_ui, cmap2pixmap, mpl_to_qt4_color, qt4_to_mpl_color
from glue.utils.qt.widget_properties import CurrentComboProperty, ValueProperty, connect_value, connect_current_combo
from glue.icons.qt import POINT_ICONS, symbol_icon


class ScatterLayerStyleWidget(QtGui.QWidget):

    size = ValueProperty('ui.value_size')
    symbol = CurrentComboProperty('ui.combo_symbol')
    alpha = ValueProperty('ui.slider_alpha')

    def __init__(self, layer_artist):

        super(ScatterLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._setup_symbol_combo()
        self._setup_color_label()

        self.layer = layer_artist.layer
        self.set_color(mpl_to_qt4_color(self.layer.style.color))
        connect_value(self.layer.style, 'alpha', self.ui.slider_alpha, value_range=(0, 1))
        connect_current_combo(self.layer.style, 'marker', self.ui.combo_symbol)
        connect_value(self.layer.style, 'markersize', self.ui.value_size)

    def _setup_color_label(self):
        self.label_color.mousePressed.connect(self.query_color)

    def _setup_symbol_combo(self):
        self._symbols = list(POINT_ICONS.keys())
        for idx, symbol in enumerate(self._symbols):
            icon = symbol_icon(symbol)
            self.combo_symbol.addItem(icon, '', userData=symbol)
        self.combo_symbol.currentIndexChanged.connect(lambda x: self.set_color(self._color))
        self.combo_symbol.setIconSize(QtCore.QSize(20, 20))
        self.combo_symbol.setMinimumSize(10, 32)

    def query_color(self, *args):
        color = QtGui.QColorDialog.getColor(self._color, self.label_color)
        if color.isValid():
            self.set_color(color)

    def set_color(self, color):
        self._color = color
        
        im = QtGui.QImage(80, 20, QtGui.QImage.Format_RGB32)
        im.fill(color)
        pm = QtGui.QPixmap.fromImage(im)
        self.label_color.setPixmap(pm)
        self.layer.style.color = qt4_to_mpl_color(self._color)


if __name__ == "__main__":
    app = get_qapp()
    options = ScatterLayerStyleWidget()
    options.show()
    app.exec_()
