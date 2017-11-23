from __future__ import absolute_import, division, print_function

import os
from collections import deque

from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt

from glue.core.parse import InvalidTagError, ParsedCommand, TAG_RE
from glue.utils.qt import load_ui, CompletionTextEdit

__all__ = ['EquationEditorDialog']


class ColorizedCompletionTextEdit(CompletionTextEdit):

    updated = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super(ColorizedCompletionTextEdit, self).__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignLeft)
        self.setUndoRedoEnabled(False)
        self._undo_stack = deque(maxlen=100)
        self._redo_stack = deque(maxlen=100)

    def insertPlainText(self, *args):
        super(ColorizedCompletionTextEdit, self).insertPlainText(*args)
        self.reformat_text()
        self.updated.emit()
        self.setAlignment(Qt.AlignLeft)

    def keyReleaseEvent(self, event):
        super(ColorizedCompletionTextEdit, self).keyReleaseEvent(event)
        self.reformat_text()
        self.updated.emit()

    def keyPressEvent(self, event):
        super(ColorizedCompletionTextEdit, self).keyPressEvent(event)
        # NOTE: We use == here instead of & for the modifiers because we don't
        # want to catch e.g. control-shift-z or other combinations.
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            if len(self._undo_stack) > 1:
                self._undo_stack.pop()
                self.setHtml(self._undo_stack[-1])
                text = self.toPlainText()
                tc = self.textCursor()
                tc.setPosition(len(text))
                self.setTextCursor(tc)
                self._cache = self._undo_stack[-1]
                self.updated.emit()

    def reformat_text(self):

        # If the text hasn't changed, no need to reformat
        if self.toPlainText() == getattr(self, '_cache', None):
            return

        # Here every time a key is released, we re-colorize the expression.
        # We show valid components in blue, and invalid ones in red. We
        # recognized components because they contain a ":" which is not valid
        # Python syntax (except if one considers lambda functions, but we can
        # probably ignore that here)
        text = self.toPlainText()

        def format_components(m):
            component = m.group(0)
            if component in self.word_list:
                return "<font color='#0072B2'><b>" + component + "</b></font>"
            else:
                return "<font color='#D55E00'><b>" + component + "</b></font>"

        html = TAG_RE.sub(format_components, text)

        tc = self.textCursor()
        pos = tc.position()

        self._undo_stack.append(html)
        self.setHtml(html)

        # Sometimes the HTML gets rid of double spaces so we have to make
        # sure the position isn't greater than the text length.
        text = self.toPlainText()
        pos = min(pos, len(text))

        tc.setPosition(pos)
        self.setTextCursor(tc)

        self._cache = self.toPlainText()


class EquationEditorDialog(QtWidgets.QDialog):

    def __init__(self, data=None, equation=None, parent=None):

        super(EquationEditorDialog, self).__init__(parent=parent)

        self.ui = load_ui('equation_editor.ui', self,
                          directory=os.path.dirname(__file__))

        self.data = data
        self.equation = equation

        # Populate data combo
        self._labels = {}
        self._components = {}
        for cid in self.data.primary_components:
            self.ui.combosel_component.addItem(cid.label, userData=cid)
            self._labels['{' + cid.label + '}'] = cid
            self._components[cid.label] = cid

        self.ui.expression.set_word_list(list(self._labels.keys()))

        self.ui.expression.insertPlainText(equation)

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

        self.ui.button_insert.clicked.connect(self._insert_component)

        self.ui.expression.updated.connect(self._update_status)
        self._update_status()

    def _insert_component(self):
        label = self.ui.combosel_component.currentText()
        self.expression.insertPlainText('{' + label + '}')

    def _update_status(self):

        # If the text hasn't changed, no need to check again
        if hasattr(self, '_cache') and self.ui.expression.toPlainText() == self._cache:
            return

        if str(self.ui.expression.toPlainText()) == "":
            self.ui.label_status.setText("")
            self.ui.button_ok.setEnabled(False)
        else:
            try:
                pc = self._get_parsed_command()
                pc.evaluate_test()
            except SyntaxError:
                self.ui.label_status.setStyleSheet('color: red')
                self.ui.label_status.setText("Incomplete or invalid syntax")
                self.ui.button_ok.setEnabled(False)
            except InvalidTagError as exc:
                self.ui.label_status.setStyleSheet('color: red')
                self.ui.label_status.setText("Invalid component: {0}".format(exc.tag))
                self.ui.button_ok.setEnabled(False)
            except Exception as exc:
                self.ui.label_status.setStyleSheet('color: red')
                self.ui.label_status.setText(str(exc))
                self.ui.button_ok.setEnabled(False)
            else:
                self.ui.label_status.setStyleSheet('color: green')
                self.ui.label_status.setText("Valid expression")
                self.ui.button_ok.setEnabled(True)

        self._cache = self.ui.expression.toPlainText()

    def _get_parsed_command(self):
        expression = str(self.ui.expression.toPlainText())
        return ParsedCommand(expression, self._components)

    def accept(self):
        self.final_expression = str(self.ui.expression.toPlainText())
        super(EquationEditorDialog, self).accept()

    def reject(self):
        self.final_expression = None
        super(EquationEditorDialog, self).reject()


if __name__ == "__main__":  # pragma: nocover

    from glue.utils.qt import get_qapp

    app = get_qapp()

    from glue.core.data import Data
    d = Data(label='test1', x=[1, 2, 3], y=[2, 3, 4], z=[3, 4, 5])
    widget = EquationEditorDialog(d, '')
    widget.exec_()
