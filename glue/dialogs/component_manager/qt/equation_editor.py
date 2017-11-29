from __future__ import absolute_import, division, print_function

import os
from collections import deque, OrderedDict

try:
    from inspect import getfullargspec
except ImportError:  # Python 2.7
    from inspect import getargspec as getfullargspec

from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt

from glue.config import link_function, link_helper
from glue.core.parse import InvalidTagError, ParsedCommand, TAG_RE
from glue.utils.qt import load_ui, CompletionTextEdit, update_combobox, fix_tab_widget_fontsize

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


def get_function_name(item):
    if hasattr(item, 'display') and item.display is not None:
        return item.display
    else:
        return item.__name__


def function_label(function):
    """ Provide a label for a function

    :param function: A member from the glue.config.link_function registry
    """
    args = getfullargspec(function.function)[0]
    args = ', '.join(args)
    output = function.output_labels
    output = ', '.join(output)
    label = "Link from %s to %s" % (args, output)
    return label


def helper_label(helper):
    """ Provide a label for a link helper

    :param helper: A member from the glue.config.link_helper registry
    """
    return helper.info


class EquationEditorDialog(QtWidgets.QDialog):

    def __init__(self, data=None, equation=None, references=None, parent=None):

        super(EquationEditorDialog, self).__init__(parent=parent)

        self.ui = load_ui('equation_editor.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tab)

        self._setup_freeform_tab(data=data, equation=equation, references=references)
        self._setup_predefined_tab(data=data)

        self.ui.tab.currentChanged.connect(self._update_status)

    def _setup_predefined_tab(self, data=None):

        # Populate category combo
        f = [f for f in link_function.members if len(f.output_labels) == 1]
        categories = sorted(set(l.category for l in f + link_helper.members))
        for category in categories:
            self.ui.combosel_category.addItem(category)
        self.ui.combosel_category.setCurrentIndex(0)
        self.ui.combosel_category.currentIndexChanged.connect(self._populate_function_combo)
        self._populate_function_combo()

        self.ui.combosel_function.setCurrentIndex(0)
        self.ui.combosel_function.currentIndexChanged.connect(self._setup_inputs)
        self._setup_inputs()

    @property
    def category(self):
        return self.ui.combosel_category.currentText()

    @property
    def function(self):
        return self.ui.combosel_function.currentData()

    @property
    def is_helper(self):
        return self.function is not None and type(self.function).__name__ == 'LinkHelper'

    @property
    def is_function(self):
        return self.function is not None and type(self.function).__name__ == 'LinkFunction'

    def _setup_inputs(self, event=None):

        if self.is_function:
            label = function_label(self.function)
            input_labels = getfullargspec(self.function.function)[0]

        else:
            label = helper_label(self.function)
            input_labels = self.function.input_labels

        self.ui.label_info.setText(label)

        self._clear_input_output_layouts()

        input_message = "The function above takes the following input(s):"

        if len(input_labels) > 1:
            input_message = input_message.replace('(s)', 's')
        else:
            input_message = input_message.replace('(s)', '')

        self.ui.layout_inout.addWidget(QtWidgets.QLabel(input_message), 0, 1, 1, 3)

        spacer1 = QtWidgets.QSpacerItem(10, 5,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Fixed)
        spacer2 = QtWidgets.QSpacerItem(10, 5,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Fixed)
        self.ui.layout_inout.addItem(spacer1, 0, 0)
        self.ui.layout_inout.addItem(spacer2, 0, 4)

        row = 0
        for a in input_labels:
            row += 1
            self._add_input_widget(a, row)

        output_message = "This function produces the following output(s) - you can set the label(s) here:"

        if len(self.function.output_labels) > 1:
            output_message = output_message.replace('(s)', 's')
        else:
            output_message = output_message.replace('(s)', '')

        row += 1
        self.ui.layout_inout.addWidget(QtWidgets.QLabel(output_message), row, 1, 1, 3)

        for a in self.function.output_labels:
            row += 1
            self._add_output_widget(a, row)

    def _clear_input_output_layouts(self):

        for row in range(self.ui.layout_inout.rowCount()):
            for col in range(self.ui.layout_inout.columnCount()):
                item = self.ui.layout_inout.itemAtPosition(row, col)
                if item is not None:
                    self.ui.layout_inout.removeItem(item)
                    if item.widget() is not None:
                        item.widget().setParent(None)

    def _add_input_widget(self, name, row):
        label = QtWidgets.QLabel(name)
        combo = QtWidgets.QComboBox()
        update_combobox(combo, list(self.references.items()))
        self.ui.layout_inout.addWidget(label, row, 1)
        self.ui.layout_inout.addWidget(combo, row, 2)

    def _add_output_widget(self, name, row):
        label = QtWidgets.QLabel(name)
        edit = QtWidgets.QLineEdit()
        self.ui.layout_inout.addWidget(label, row, 1)
        self.ui.layout_inout.addWidget(edit, row, 2)

    def _populate_function_combo(self, event=None):
        """
        Add name of functions to function combo box
        """
        f = [f for f in link_function.members if len(f.output_labels) == 1]
        functions = ((get_function_name(l[0]), l) for l in f + link_helper.members if l.category == self.category)
        update_combobox(self.ui.combosel_function, functions)
        self._setup_inputs()

    def _setup_freeform_tab(self, data=None, equation=None, references=None):

        self.equation = equation

        # Get mapping from label to component ID
        if references is not None:
            self.references = references
        elif data is not None:
            self.references = OrderedDict()
            for cid in data.primary_components:
                self.references[cid.label] = cid

        # Populate component combo
        for label, cid in self.references.items():
            self.ui.combosel_component.addItem(label, userData=cid)

        # Set up labels for auto-completion
        labels = ['{' + label + '}' for label in self.references]
        self.ui.expression.set_word_list(labels)

        self.ui.expression.insertPlainText(equation)

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

        self.ui.button_insert.clicked.connect(self._insert_component)

        self.ui.expression.updated.connect(self._update_status)
        self._update_status()

    def _insert_component(self):
        label = self.ui.combosel_component.currentText()
        self.expression.insertPlainText('{' + label + '}')

    def _update_status(self, event=None):

        if self.ui.tab.currentIndex() == 0:

            self.ui.label_status.setStyleSheet('color: green')
            self.ui.label_status.setText('')
            self.ui.button_ok.setEnabled(True)

        else:

            # If the text hasn't changed, no need to check again
            if hasattr(self, '_cache') and self._get_raw_command() == self._cache and event is None:
                return

            if self._get_raw_command() == "":
                self.ui.label_status.setText("")
                self.ui.button_ok.setEnabled(False)
            else:
                try:
                    pc = self._get_parsed_command()
                    result = pc.evaluate_test()
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
                    if result is None:
                        self.ui.label_status.setStyleSheet('color: red')
                        self.ui.label_status.setText("Expression should not return None")
                        self.ui.button_ok.setEnabled(False)
                    else:
                        self.ui.label_status.setStyleSheet('color: green')
                        self.ui.label_status.setText("Valid expression")
                        self.ui.button_ok.setEnabled(True)

            self._cache = self._get_raw_command()

    def _get_raw_command(self):
        return str(self.ui.expression.toPlainText())

    def _get_parsed_command(self):
        expression = self._get_raw_command()
        return ParsedCommand(expression, self.references)

    def accept(self):
        self.final_expression = self._get_parsed_command()._cmd
        super(EquationEditorDialog, self).accept()

    def reject(self):
        self.final_expression = None
        super(EquationEditorDialog, self).reject()


if __name__ == "__main__":  # pragma: nocover

    from glue.main import load_plugins
    from glue.utils.qt import get_qapp

    app = get_qapp()
    load_plugins()

    from glue.core.data import Data
    d = Data(label='test1', x=[1, 2, 3], y=[2, 3, 4], z=[3, 4, 5])
    widget = EquationEditorDialog(d, '')
    widget.exec_()
