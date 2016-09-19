import os

from qtpy import QtWidgets

from glue.utils.qt import load_ui

from glue_new_viewers.common.qt_helpers import autoconnect_qt


class HistogramLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(HistogramLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        # TODO: In future, should pass only state not layer?
        self.layer_state = layer.layer_state

        connect_kwargs = {
            'alpha': dict(value_range=(0, 1))
        }

        autoconnect_qt(self.layer_state, self.ui, connect_kwargs)
