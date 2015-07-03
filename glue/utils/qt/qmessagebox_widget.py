# A patched version of QMessageBox that allows copying the error

import os
from ...external.qt import QtGui

__all__ = ['QMessageBoxPatched']


class QMessageBoxPatched(QtGui.QMessageBox):

    def __init__(self, *args, **kwargs):

        super(QMessageBoxPatched, self).__init__(*args, **kwargs)

        copy_action = QtGui.QAction('&Copy', self)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_detailed)

        select_all = QtGui.QAction('Select &All', self)
        select_all.setShortcut(QtGui.QKeySequence.SelectAll)
        select_all.triggered.connect(self.select_all)

        menubar = QtGui.QMenuBar()
        editMenu = menubar.addMenu('&Edit')
        editMenu.addAction(copy_action)
        editMenu.addAction(select_all)

        self.layout().setMenuBar(menubar)

    @property
    def detailed_text_widget(self):
        return self.findChild(QtGui.QTextEdit)

    def select_all(self):
        self.detailed_text_widget.selectAll()

    def copy_detailed(self):
        clipboard = QtGui.QApplication.clipboard()
        selected_text = self.detailed_text_widget.textCursor().selectedText()
        # Newlines are unicode, so need to normalize them to ASCII
        selected_text = os.linesep.join(selected_text.splitlines())
        clipboard.setText(selected_text)
