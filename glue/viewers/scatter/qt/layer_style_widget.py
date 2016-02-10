from __future__ import absolute_import, division, print_function

import os
from time import ctime

from glue.external.qt import QtCore, QtGui, get_qapp
from glue import core
from glue import config
from glue.utils.qt import load_ui, cmap2pixmap
from glue.utils.qt.widget_properties import CurrentComboTextProperty, ValueProperty
from glue.icons.qt import POINT_ICONS, symbol_icon


class ColorWidget(QtGui.QLabel):
    mousePressed = QtCore.Signal()

    def mousePressEvent(self, event):
        self.mousePressed.emit()
        event.accept()


class ScatterLayerStyleWidget(QtGui.QWidget):

    size = ValueProperty('ui.value_size')
    symbol = CurrentComboTextProperty('ui.combo_symbol')
    alpha = ValueProperty('ui.slider_alpha')

    def __init__(self):

        super(ScatterLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._setup_symbol_combo()
        self._setup_color_label()

    def _setup_color_label(self):
        self.set_color(QtGui.QColor(0, 0, 0))
        self.label_color.mousePressed.connect(self.query_color)

    def set_color(self, color):
        self._color = color
        pm = symbol_icon(self.symbol(), color).pixmap(30, 30)
        self.label_color.setPixmap(pm)

    def symbol(self):
        return self._symbols[self.combo_symbol.currentIndex()]

    def query_color(self, *args):
        color = QtGui.QColorDialog.getColor(self._color, self.label_color)
        if color.isValid():
            self.set_color(color)

    def _setup_symbol_combo(self):
        self._symbols = list(POINT_ICONS.keys())
        for idx, symbol in enumerate(self._symbols):
            icon = symbol_icon(symbol)
            self.combo_symbol.addItem(icon, '')
        self.combo_symbol.setIconSize(QtCore.QSize(10, 10))
        # self.combo_symbol.setMinimumSize(10, 32)
        self.combo_symbol.currentIndexChanged.connect(
            lambda x: self.set_color(self.color()))
        self.combo_symbol.setIconSize(QtCore.QSize(20, 20))
        self.combo_symbol.setMinimumSize(10, 32)

    def color(self):
        return self._color

    def _query_color(self):
        color = QtGui.QColorDialog.getColor(self._color, parent=self)
        self._set_color(color)

    def _set_color(self, color):
        self._color = color
        self.palette = QtGui.QPalette()
        self.palette.setColor(QtGui.QPalette.Button, color)
        self.button_color.setPalette(self.palette)



if __name__ == "__main__":
    app = get_qapp()
    options = ScatterLayerStyleWidget()
    options.show()
    app.exec_()
