import os

from qtpy import QtWidgets

from echo.qt import autoconnect_callbacks_to_qt
from glue.utils.qt import load_ui

__all__ = ['DendrogramOptionsWidget']


class DendrogramOptionsWidget(QtWidgets.QWidget):

    def __init__(self, viewer_state, session, parent=None):

        super(DendrogramOptionsWidget, self).__init__(parent=parent)

        self.ui = load_ui('options_widget.ui', self,
                          directory=os.path.dirname(__file__))

        self._connections = autoconnect_callbacks_to_qt(viewer_state, self.ui)

        self.viewer_state = viewer_state

    def reset_limits(self):
        self.viewer_state.reset_limits()
