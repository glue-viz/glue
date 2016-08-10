import os

from qtpy import QtWidgets

from glue.utils.qt.widget_properties import CurrentComboDataProperty
from glue.utils.qt import load_ui, update_combobox

__all__ = ["LayerWidget"]


class LayerWidget(QtWidgets.QWidget):

    layer = CurrentComboDataProperty('ui.combo_active_layer')

    def __init__(self, parent=None, data_viewer=None):

        super(LayerWidget, self).__init__(parent=parent)

        self.ui = load_ui('layer_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._layers = []

    def __contains__(self, layer):
        return layer in self._layers

    def add_layer(self, layer):
        self._layers.append(layer)
        self._update_combobox()

    def remove_layer(self, layer):
        self._layers.remove(layer)
        self._update_combobox()

    def _update_combobox(self):
        labeldata = [(layer.label, layer) for layer in self._layers]
        update_combobox(self.ui.combo_active_layer, labeldata)
