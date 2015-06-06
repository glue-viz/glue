# Code adapted from:
#
# http://rowinggolfer.blogspot.de/2010/08/qtextedit-with-autocompletion-using.html
#
# and based on:
#
# http://qt-project.org/doc/qt-4.8/tools-customcompleter.html

from ...external.qt import QtGui, QtCore


__all__ = ["CompletionTextEdit"]


class CompletionTextEdit(QtGui.QTextEdit):

    def __init__(self, parent=None):

        super(CompletionTextEdit, self).__init__(parent)

        self.setMinimumWidth(400)
        self.completer = None
        self.word_list = None

        self.moveCursor(QtGui.QTextCursor.End)

    def set_word_list(self, word_list):
        self.word_list = word_list
        self.set_completer(QtGui.QCompleter(word_list))

    def set_completer(self, completer):

        if self.completer:
            self.disconnect(self.completer, 0, self, 0)
        if not completer:
            return

        self.completer = completer

        self.completer.setWidget(self)
        self.completer.setCompletionMode(QtGui.QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer.activated.connect(self.insert_completion)

    def insert_completion(self, completion):

        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        tc.deleteChar()

        completion = completion + " "

        self.setTextCursor(tc)

        self.insertPlainText(completion)

    def text_under_cursor(self):
        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        return tc.selectedText()

    # The following methods override methods in QTextEdit and should not be
    # renamed.

    def focusInEvent(self, event):
        if self.completer:
            self.completer.setWidget(self)
        QtGui.QTextEdit.focusInEvent(self, event)

    def keyPressEvent(self, event):

        if self.completer and self.completer.popup().isVisible():
            if event.key() in (
                    QtCore.Qt.Key_Enter,
                    QtCore.Qt.Key_Return,
                    QtCore.Qt.Key_Escape,
                    QtCore.Qt.Key_Tab,
                    QtCore.Qt.Key_Backtab):
                event.ignore()
                return

        # Check if TAB has been pressed
        is_shortcut = event.key() == QtCore.Qt.Key_Tab

        if not self.completer or not is_shortcut:
            QtGui.QTextEdit.keyPressEvent(self, event)
            return

        eow = "~!@#$%^&*()_+{}|:\"<>?,./;'[]\\-="

        completion_prefix = self.text_under_cursor()

        if not is_shortcut and (len(event.text()) == 0 or event.text()[-1:] in eow):
            self.completer.popup().hide()
            return

        if (completion_prefix != self.completer.completionPrefix()):
            self.completer.setCompletionPrefix(completion_prefix)
            popup = self.completer.popup()
            popup.setCurrentIndex(self.completer.completionModel().index(0, 0))

        cr = self.cursorRect()
        cr.setWidth(self.completer.popup().sizeHintForColumn(0) +
                    self.completer.popup().verticalScrollBar().sizeHint().width())
        self.completer.complete(cr)
