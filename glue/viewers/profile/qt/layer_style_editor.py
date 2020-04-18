import os

from qtpy import QtWidgets

from echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui


class ProfileLayerStyleEditor(QtWidgets.QWidget):

    def __init__(self, layer, parent=None):

        super(ProfileLayerStyleEditor, self).__init__(parent=parent)

        self.ui = load_ui('layer_style_editor.ui', self,
                          directory=os.path.dirname(__file__))

        connect_kwargs = {'alpha': dict(value_range=(0, 1))}

        self._connections = autoconnect_callbacks_to_qt(layer.state, self.ui, connect_kwargs)

        self.viewer_state = layer.state.viewer_state
        self.viewer_state.add_callback('normalize', self._on_normalize_change)
        self._on_normalize_change()

    def _on_normalize_change(self, *event):
        self.ui.label_limits.setVisible(self.viewer_state.normalize)
        self.ui.valuetext_v_min.setVisible(self.viewer_state.normalize)
        self.ui.valuetext_v_max.setVisible(self.viewer_state.normalize)
        self.ui.button_flip_limits.setVisible(self.viewer_state.normalize)
