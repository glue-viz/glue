from __future__ import absolute_import, division, print_function

import os

from glue.external.qt import QtGui
from glue.utils.qt import load_ui, mpl_to_qt4_color, qt4_to_mpl_color
from glue.utils.qt.widget_properties import ValueProperty, connect_value


class HistogramLayerStyleWidget(QtGui.QWidget):

    alpha = ValueProperty('ui.slider_alpha')

    def __init__(self, layer_artist):

        super(HistogramLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._setup_color_label()

        self.layer = layer_artist.layer
        self.set_color(mpl_to_qt4_color(self.layer.style.color))
        connect_value(self.layer.style, 'alpha', self.ui.slider_alpha, scaling=1./100.)

    def _setup_color_label(self):
        self.label_color.mousePressed.connect(self.query_color)

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

