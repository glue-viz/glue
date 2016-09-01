from __future__ import absolute_import, division, print_function

from qtpy.QtCore import Qt
from qtpy import QtCore, QtWidgets
from glue.icons.qt import POINT_ICONS, symbol_icon
from glue.utils.qt import mpl_to_qt4_color, qt4_to_mpl_color


class ColorWidget(QtWidgets.QLabel):
    mousePressed = QtCore.Signal()

    def mousePressEvent(self, event):
        self.mousePressed.emit()
        event.accept()


class StyleDialog(QtWidgets.QDialog):

    """Dialog which edits the style of a layer (Data or Subset)

    Use via StyleDialog.edit_style(layer)
    """

    def __init__(self, layer, parent=None, edit_label=True):
        super(StyleDialog, self).__init__(parent)
        self.setWindowTitle("Style Editor")
        self.layer = layer
        self._edit_label = edit_label
        self._symbols = list(POINT_ICONS.keys())

        self._setup_widgets()
        self._connect()

    def _setup_widgets(self):
        self.layout = QtWidgets.QFormLayout()

        self.size_widget = QtWidgets.QSpinBox()
        self.size_widget.setMinimum(1)
        self.size_widget.setMaximum(40)
        self.size_widget.setValue(self.layer.style.markersize)

        self.label_widget = QtWidgets.QLineEdit()
        self.label_widget.setText(self.layer.label)
        self.label_widget.selectAll()

        self.symbol_widget = QtWidgets.QComboBox()
        for idx, symbol in enumerate(self._symbols):
            icon = symbol_icon(symbol)
            self.symbol_widget.addItem(icon, '')
            if symbol is self.layer.style.marker:
                self.symbol_widget.setCurrentIndex(idx)
        self.symbol_widget.setIconSize(QtCore.QSize(20, 20))
        self.symbol_widget.setMinimumSize(10, 32)

        self.color_widget = ColorWidget()
        self.color_widget.setStyleSheet('ColorWidget {border: 1px solid;}')
        color = self.layer.style.color
        color = mpl_to_qt4_color(color, alpha=self.layer.style.alpha)
        self.set_color(color)

        self.okcancel = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                               QtWidgets.QDialogButtonBox.Cancel)

        if self._edit_label:
            self.layout.addRow("Label", self.label_widget)
        self.layout.addRow("Symbol", self.symbol_widget)
        self.layout.addRow("Color", self.color_widget)
        self.layout.addRow("Size", self.size_widget)

        self.layout.addWidget(self.okcancel)

        self.setLayout(self.layout)
        self.layout.setContentsMargins(6, 6, 6, 6)

    def _connect(self):
        self.color_widget.mousePressed.connect(self.query_color)
        self.symbol_widget.currentIndexChanged.connect(
            lambda x: self.set_color(self.color()))
        self.okcancel.accepted.connect(self.accept)
        self.okcancel.rejected.connect(self.reject)
        self.setFocusPolicy(Qt.StrongFocus)

    def query_color(self, *args):
        color = QtWidgets.QColorDialog.getColor(self._color, self.color_widget,
                                            "",
                                            QtWidgets.QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self.set_color(color)

    def color(self):
        return self._color

    def set_color(self, color):
        self._color = color
        pm = symbol_icon(self.symbol(), color).pixmap(30, 30)
        self.color_widget.setPixmap(pm)

    def size(self):
        return self.size_widget.value()

    def label(self):
        return str(self.label_widget.text())

    def symbol(self):
        return self._symbols[self.symbol_widget.currentIndex()]

    def update_style(self):
        if self._edit_label:
            self.layer.label = self.label()
        self.layer.style.color = qt4_to_mpl_color(self.color())
        self.layer.style.alpha = self.color().alpha() / 255.
        self.layer.style.marker = self.symbol()
        self.layer.style.markersize = self.size()

    @classmethod
    def edit_style(cls, layer):
        self = cls(layer)
        result = self.exec_()

        if result == self.Accepted:
            self.update_style()

    @classmethod
    def dropdown_editor(cls, item, pos, **kwargs):
        """
        Create a dropdown-style modal editor to edit the style of a
        given item

        :param item: Item with a .label and .style to edit
        :param pos: A QPoint to anchor the top-left corner of the dropdown at
        :param kwargs: Extra keywords to pass to StyleDialogs's constructor
        """
        self = cls(item, **kwargs)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)

        pos = self.mapFromGlobal(pos)
        self.move(pos)
        if self.exec_() == self.Accepted:
            self.update_style()


if __name__ == "__main__":

    from glue.core import Data

    d = Data(label='data label', x=[1, 2, 3, 4])
    StyleDialog.edit_style(d)

    print("New layer properties")
    print(d.label)
    print('color: ', d.style.color)
    print('marker: ', d.style.marker)
    print('marker size: ', d.style.markersize)
    print('alpha ', d.style.alpha)
