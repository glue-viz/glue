# Code adapted from:
#
# http://rowinggolfer.blogspot.de/2010/08/qtextedit-with-autocompletion-using.html
#
# and based on:
#
# http://qt-project.org/doc/qt-4.8/tools-customcompleter.html

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt


__all__ = ["CompletionTextEdit"]


class CompletionTextEdit(QtWidgets.QTextEdit):

    def __init__(self, parent=None):

        super(CompletionTextEdit, self).__init__(parent)

        self.setMinimumWidth(400)
        self.completer = None
        self.word_list = None

        self.moveCursor(QtGui.QTextCursor.End)

    def set_word_list(self, word_list):
        self.word_list = word_list
        self.set_completer(QtWidgets.QCompleter(word_list))

    def set_completer(self, completer):

        if self.completer:
            self.disconnect(self.completer, 0, self, 0)
        if not completer:
            return

        self.completer = completer

        self.completer.setWidget(self)
        self.completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.activated.connect(self.insert_completion)

    def insert_completion(self, completion):

        tc = self.textCursor()

        existing = self.text_under_cursor()

        completion = completion[len(existing):] + " "

        self.setTextCursor(tc)

        self.insertPlainText(completion)

        self.completer.setCompletionPrefix('')

    def text_under_cursor(self):
        tc = self.textCursor()
        text = self.toPlainText()
        pos1 = tc.position()
        if pos1 == 0 or text[pos1 - 1] == ' ':
            return ''
        else:
            sub = text[:pos1]
            if ' ' in sub:
                pos2 = sub.rindex(' ')
                return sub[pos2 + 1:]
            else:
                return sub

    # The following methods override methods in QTextEdit and should not be
    # renamed.

    def focusInEvent(self, event):
        if self.completer:
            self.completer.setWidget(self)
        QtWidgets.QTextEdit.focusInEvent(self, event)

    def keyPressEvent(self, event):

        if self.completer and self.completer.popup().isVisible():
            if event.key() in (
                    Qt.Key_Enter,
                    Qt.Key_Return,
                    Qt.Key_Escape,
                    Qt.Key_Tab,
                    Qt.Key_Backtab):
                event.ignore()
                return

        # Check if TAB has been pressed
        is_shortcut = event.key() == Qt.Key_Tab

        if not self.completer or not is_shortcut:
            QtWidgets.QTextEdit.keyPressEvent(self, event)
            return

        eow = "~!@#$%^&*()_+{}|:\"<>?,./;'[]\\-="

        completion_prefix = self.text_under_cursor()

        if not is_shortcut and (len(event.text()) == 0 or event.text()[-1:] in eow):
            self.completer.popup().hide()
            return

        self.completer.setCompletionPrefix(completion_prefix)
        popup = self.completer.popup()
        popup.setCurrentIndex(self.completer.completionModel().index(0, 0))

        cr = self.cursorRect()
        cr.setWidth(self.completer.popup().sizeHintForColumn(0) +
                    self.completer.popup().verticalScrollBar().sizeHint().width())
        self.completer.complete(cr)
