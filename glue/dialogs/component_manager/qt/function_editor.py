from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

try:
    from inspect import getfullargspec
except ImportError:  # Python 2.7
    from inspect import getargspec as getfullargspec

from qtpy import QtWidgets

from glue.config import link_function, link_helper
from glue.utils.qt import load_ui, update_combobox

__all__ = ['FunctionEditorDialog']


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


class FunctionEditorDialog(QtWidgets.QDialog):

    def __init__(self, data=None, references=None, parent=None):

        super(FunctionEditorDialog, self).__init__(parent=parent)

        self.ui = load_ui('function_editor.ui', self,
                          directory=os.path.dirname(__file__))

        # Get mapping from label to component ID
        if references is not None:
            self.references = references
        elif data is not None:
            self.references = OrderedDict()
            for cid in data.primary_components:
                self.references[cid.label] = cid

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


if __name__ == "__main__":  # pragma: nocover

    from glue.main import load_plugins
    from glue.utils.qt import get_qapp

    app = get_qapp()
    load_plugins()

    from glue.core.data import Data
    d = Data(label='test1', x=[1, 2, 3], y=[2, 3, 4], z=[3, 4, 5])
    widget = FunctionEditorDialog(d)
    widget.exec_()
