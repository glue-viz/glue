from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets
from glue.utils.qt import load_ui, connect_color
from glue.utils.qt.widget_properties import ValueProperty, connect_value


class HistogramLayerStyleWidget(QtWidgets.QWidget):

    alpha = ValueProperty('ui.slider_alpha', value_range=(0, 1))

    def __init__(self, layer_artist):

        super(HistogramLayerStyleWidget, self).__init__()

        self.ui = load_ui('layer_style_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self.layer = layer_artist.layer

        # Set up connections
        self._connect_global()

        # Set initial values
        self.ui.label_color.setColor(self.layer.style.color)
        self.alpha = self.layer.style.alpha

    def _connect_global(self):
        connect_color(self.layer.style, 'color', self.ui.label_color)
        connect_value(self.layer.style, 'alpha', self.ui.slider_alpha, value_range=(0, 1))
