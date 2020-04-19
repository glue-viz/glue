import os

from qtpy import QtWidgets

from glue.utils.qt import load_ui

__all__ = ['LegendEditorWidget']


class LegendEditorWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(LegendEditorWidget, self).__init__(parent=parent)
        self.ui = load_ui('legend_editor.ui', self,
                          directory=os.path.dirname(__file__))
