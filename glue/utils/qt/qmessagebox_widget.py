# A patched version of QMessageBox that allows copying the error

from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets, QtGui

__all__ = ['QMessageBoxPatched']


class QMessageBoxPatched(QtWidgets.QMessageBox):

    def __init__(self, *args, **kwargs):

        super(QMessageBoxPatched, self).__init__(*args, **kwargs)

        copy_action = QtWidgets.QAction('&Copy', self)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_detailed)

        select_all = QtWidgets.QAction('Select &All', self)
        select_all.setShortcut(QtGui.QKeySequence.SelectAll)
        select_all.triggered.connect(self.select_all)

        menubar = QtWidgets.QMenuBar()
        editMenu = menubar.addMenu('&Edit')
        editMenu.addAction(copy_action)
        editMenu.addAction(select_all)

        self.layout().setMenuBar(menubar)

    @property
    def detailed_text_widget(self):
        return self.findChild(QtWidgets.QTextEdit)

    def select_all(self):
        self.detailed_text_widget.selectAll()

    def copy_detailed(self):
        clipboard = QtWidgets.QApplication.clipboard()
        selected_text = self.detailed_text_widget.textCursor().selectedText()
        # Newlines are unicode, so need to normalize them to ASCII
        selected_text = os.linesep.join(selected_text.splitlines())
        clipboard.setText(selected_text)
