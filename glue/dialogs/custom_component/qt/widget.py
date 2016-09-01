from __future__ import absolute_import, division, print_function

import os
import re

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from glue.core import parse
from glue import core
from glue.utils import nonpartial
from glue.utils.qt import load_ui
from glue.utils.qt import CompletionTextEdit

__all__ = ['CustomComponentWidget']


def disambiguate(label, labels):
    """ Changes name of label if it conflicts with labels list

    Parameters
    ----------
    label : str
        The label to change the name of
    labels : iterable
        A list of all labels

    Returns
    -------
    label : str
        If needed, appended with a suffix "_{number}". The output does not
        appear in labels
    """
    label = label.replace(' ', '_')
    if label not in labels:
        return label
    suffix = 1
    while label + ('_%i' % suffix) in labels:
        suffix += 1
    return label + ('_%i' % suffix)


class ColorizedCompletionTextEdit(CompletionTextEdit):

    updated = QtCore.Signal()

    def insertPlainText(self, *args):
        super(ColorizedCompletionTextEdit, self).insertPlainText(*args)
        self.reformat_text()
        self.updated.emit()

    def keyReleaseEvent(self, event):
        super(ColorizedCompletionTextEdit, self).keyReleaseEvent(event)
        self.reformat_text()
        self.updated.emit()

    def reformat_text(self):

        # Here every time a key is released, we re-colorize the expression.
        # We show valid components in blue, and invalid ones in red. We
        # recognized components because they contain a ":" which is not valid
        # Python syntax (except if one considers lambda functions, but we can
        # probably ignore that here)
        text = self.toPlainText()

        # If there are no : in the text we don't need to do anything
        if not ":" in text:
            return

        pattern = '[^\\s]*:[^\\s]*'

        def format_components(m):
            component = m.group(0)
            if component in self.word_list:
                return "<font color='#0072B2'><b>" + component + "</b></font> "
            else:
                return "<font color='#D55E00'><b>" + component + "</b></font> "

        html = re.sub(pattern, format_components, text)

        tc = self.textCursor()
        pos = tc.position()

        self.setHtml(html)

        # Sometimes the HTML gets rid of double spaces so we have to make
        # sure the position isn't greater than the text length.
        text = self.toPlainText()
        pos = min(pos, len(text))

        tc.setPosition(pos)
        self.setTextCursor(tc)
        self.setAlignment(Qt.AlignCenter)


class CustomComponentWidget(QtWidgets.QDialog):
    """
    Dialog to add derived components to data via parsed commands.
    """

    def __init__(self, collection, parent=None):

        super(CustomComponentWidget, self).__init__(parent=parent)

        # Load in ui file to set up widget
        self.ui = load_ui('widget.ui', self,
                          directory=os.path.dirname(__file__))

        # In the ui file we do not create the text field for the expression
        # because we want to use a custom widget that supports auto-complete.
        self.ui.expression.setAlignment(Qt.AlignCenter)

        self._labels = {}
        self._data = {}
        self._collection = collection
        self._gather_components()
        self._gather_data()
        self._init_widgets()
        self._connect()

        # Set up auto-completion. While the auto-complete window is open, we
        # cannot add/remove datasets or other components, so we can populate
        # the auto_completer straight off.
        self.ui.expression.set_word_list(list(self._labels.keys()))

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

        self.ui.expression.updated.connect(self._update_status)
        self._update_status()

    def _update_status(self):
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
            except parse.InvalidTagError as exc:
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

    def _connect(self):
        cl = self.ui.component_list
        cl.itemDoubleClicked.connect(self._add_to_expression)

    def _init_widgets(self):
        """
        Set up default state of widget
        """
        comps = self.ui.component_list
        comps.addItems(sorted(self._labels.keys()))
        data = self.ui.data_list
        data.addItems(sorted(self._data.keys()))

    def _gather_components(self):
        """
        Build a mapping from unique labels -> componentIDs
        """
        comps = set()
        for data in self._collection:
            for c in data.components:
                if c in comps:
                    continue
                label = "%s:%s" % (data.label, c)
                label = disambiguate(label, self._labels)
                self._labels[label] = c
                comps.add(c)

    def _gather_data(self):
        """
        Build a mapping from unique labels -> data objects
        """
        for data in self._collection:
            label = data.label
            label = disambiguate(label, self._data)
            self._data[label] = data

    def _selected_data(self):
        """
        Yield all data objects that are selected in the DataList
        """
        for items in self.ui.data_list.selectedItems():
            yield self._data[str(items.text())]

    def _create_link(self):
        """
        Create a ComponentLink from the state of the GUI

        Returns
        -------
        A new component link
        """
        pc = self._get_parsed_command()
        label = str(self.ui.new_label.text()) or 'new component'
        new_id = core.data.ComponentID(label)
        link = parse.ParsedComponentLink(new_id, pc)
        return link

    def _get_parsed_command(self):

        expression = str(self.ui.expression.toPlainText())

        # To maintain backward compatibility with previous versions of glue,
        # we add curly brackets around the components in the expression.
        pattern = '[^\\s]*:[^\\s]*'

        def add_curly(m):
            return "{" + m.group(0) + "}"
        expression = re.sub(pattern, add_curly, expression)

        return parse.ParsedCommand(expression, self._labels)

    @property
    def _number_targets(self):
        """
        How many targets are selected
        """
        return len(self.ui.data_list.selectedItems())

    def _add_link_to_targets(self, link):
        """
        Add a link to all the selected data
        """
        for target in self._selected_data():
            target.add_component_link(link)

    def _add_to_expression(self, item):
        """
        Add a component list item to the expression editor
        """
        addition = '%s ' % item.text()
        expression = self.ui.expression
        expression.insertPlainText(addition)

    def accept(self):
        if self._number_targets == 0:
            QtWidgets.QMessageBox.critical(self.ui, "Error", "Please specify the target dataset(s)",
                                       buttons=QtWidgets.QMessageBox.Ok)
        elif len(self.ui.new_label.text()) == 0:
            QtWidgets.QMessageBox.critical(self.ui, "Error", "Please specify the new component name",
                                       buttons=QtWidgets.QMessageBox.Ok)
        else:
            link = self._create_link()
            if link:
                self._add_link_to_targets(link)
            super(CustomComponentWidget, self).accept()


def main():
    from glue.core.data import Data
    from glue.core.data_collection import DataCollection
    import numpy as np

    x = np.random.random((5, 5))
    y = x * 3
    data = DataCollection(Data(label='test', x=x, y=y))

    widget = CustomComponentWidget(data)
    widget.exec_()

    for d in data:
        print(d.label)
        for c in d.components:
            print('\t%s' % c)

if __name__ == "__main__":
    from glue.utils.qt import get_qapp
    app = get_qapp()
    main()
