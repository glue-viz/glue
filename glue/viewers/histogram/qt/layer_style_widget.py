from __future__ import absolute_import, division, print_function

import os

from glue.external.qt import QtGui
from glue.utils.qt import load_ui, connect_color
from glue.utils.qt.widget_properties import ValueProperty, connect_value


class HistogramLayerStyleWidget(QtGui.QWidget):

    alpha = ValueProperty('ui.slider_alpha')

    def __init__(self, layer_artist):

        super(HistogramLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self.layer = layer_artist.layer
        connect_color(self.layer.style, 'color', self.ui.label_color)
        connect_value(self.layer.style, 'alpha', self.ui.slider_alpha, value_range=(0, 1))
