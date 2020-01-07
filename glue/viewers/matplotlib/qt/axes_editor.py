import os

from qtpy import QtWidgets

from glue.utils.qt import load_ui

__all__ = ['AxesEditorWidget']


class AxesEditorWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(AxesEditorWidget, self).__init__(parent=parent)
        self.ui = load_ui('axes_editor.ui', self,
                          directory=os.path.dirname(__file__))
