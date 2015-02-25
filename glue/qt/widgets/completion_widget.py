# Code adapted from:
#
# http://rowinggolfer.blogspot.de/2010/08/qtextedit-with-autocompletion-using.html
#
# and based on:
#
# http://qt-project.org/doc/qt-4.8/tools-customcompleter.html

from ...external.qt import QtGui, QtCore


try:  # Python 2

    QString = QtCore.QString

except AttributeError:  # Python 3

    class QString(str):
        def isEmpty(self):
            return len(self) > 0
        def length(self):
            return len(self)
        def contains(self, sub):
            return sub in self
        def right(self, n):
            try:
                return self[-n:]
            except IndexError:
                return self


class CompletionTextEdit(QtGui.QTextEdit):

    def __init__(self, parent=None):
        super(CompletionTextEdit, self).__init__(parent)
        self.setMinimumWidth(400)
        self.completer = None
        self.moveCursor(QtGui.QTextCursor.End)

    def setCompleter(self, completer):
        if self.completer:
            self.disconnect(self.completer, 0, self, 0)
        if not completer:
            return

        completer.setWidget(self)
        completer.setCompletionMode(QtGui.QCompleter.PopupCompletion)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer = completer
        self.connect(self.completer,
            QtCore.SIGNAL("activated(const QString&)"), self.insertCompletion)

    def insertCompletion(self, completion):
        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        tc.deleteChar()
        tc.insertHtml(' <font color="blue"><b>' + completion + '</b></font> ')
        self.setTextCursor(tc)

    def textUnderCursor(self):
        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        return tc.selectedText()

    def focusInEvent(self, event):
        if self.completer:
            self.completer.setWidget(self);
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
        isShortcut = event.key() == QtCore.Qt.Key_Tab

        if (not self.completer or not isShortcut):
            QtGui.QTextEdit.keyPressEvent(self, event)
            return

        eow = QString("~!@#$%^&*()_+{}|:\"<>?,./;'[]\\-=")

        completionPrefix = self.textUnderCursor()

        if (not isShortcut and (QString(event.text()).isEmpty() or
                QString(completionPrefix).length() < 3 or
                eow.contains(QString(event.text()).right(1)))):
            self.completer.popup().hide()
            return

        if (completionPrefix != self.completer.completionPrefix()):
            self.completer.setCompletionPrefix(completionPrefix)
            popup = self.completer.popup()
            popup.setCurrentIndex(
                self.completer.completionModel().index(0,0))

        cr = self.cursorRect()
        cr.setWidth(self.completer.popup().sizeHintForColumn(0)
            + self.completer.popup().verticalScrollBar().sizeHint().width())
        self.completer.complete(cr)
