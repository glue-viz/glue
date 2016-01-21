from __future__ import absolute_import, division, print_function

import re

from glue.external.qt import QtGui

from glue import core
from glue.core import parse
from glue.utils.qt import CompletionTextEdit

from glue.qt.qtutil import load_ui



def disambiguate(label, labels):
    """ Changes name of label if it conflicts with labels list

    Parameters
    ----------
    label : string
    labels : collection of strings

    Returns
    -------
    label, perhaps appended with a suffix "_{number}". The output
    does not appear in labels
    """
    label = label.replace(' ', '_')
    if label not in labels:
        return label
    suffix = 1
    while label + ('_%i' % suffix) in labels:
        suffix += 1
    return label + ('_%i' % suffix)


class ColorizedCompletionTextEdit(CompletionTextEdit):

    def insertPlainText(self, *args):
        super(ColorizedCompletionTextEdit, self).insertPlainText(*args)
        self.reformat_text()

    def keyReleaseEvent(self, event):
        super(ColorizedCompletionTextEdit, self).keyReleaseEvent(event)
        self.reformat_text()

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



class CustomComponentWidget(object):
    """
    Dialog to add derived components to data via parsed commands.
    """

    def __init__(self, collection, parent=None):

        # Load in ui file to set up widget
        self.ui = load_ui('custom_component_widget', parent)

        # In the ui file we do not create the text field for the expression
        # because we want to use a custom widget that supports auto-complete.
        self.ui.expression = ColorizedCompletionTextEdit()
        self.ui.verticalLayout_3.addWidget(self.ui.expression)
        self.ui.expression.setAlignment(Qt.AlignCenter)
        self.ui.expression.setObjectName("expression")
        self.ui.expression.setToolTip("Define a new component. You can either "
                                      "type out the full name of a component\n"
                                      "with the data:component syntax, or "
                                      "start typing and press TAB to use "
                                      "tab-completion.\n Blue-colored "
                                      "components are valid, while "
                                      "Red-colored components are invalid.")

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

    def _connect(self):
        cl = self.ui.component_list
        cl.itemDoubleClicked.connect(self._add_to_expression)

    def _init_widgets(self):
        """ Set up default state of widget """
        comps = self.ui.component_list
        comps.addItems(sorted(self._labels.keys()))
        data = self.ui.data_list
        data.addItems(sorted(self._data.keys()))

    def _gather_components(self):
        """ Build a mapping from unique labels -> componentIDs """
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
        """ Build a mapping from unique labels -> data objects """
        for data in self._collection:
            label = data.label
            label = disambiguate(label, self._data)
            self._data[label] = data

    def _selected_data(self):
        """ Yield all data objects that are selected in the DataList """
        for items in self.ui.data_list.selectedItems():
            yield self._data[str(items.text())]

    def _create_link(self):
        """ Create a ComponentLink form the state of the GUI

        Returns
        -------
        A new component link
        """

        expression = str(self.ui.expression.toPlainText())

        # To maintain backward compatibility with previous versions of glue,
        # we add curly brackets around the components in the expression.
        pattern = '[^\\s]*:[^\\s]*'
        def add_curly(m):
            return "{" + m.group(0) + "}"
        expression = re.sub(pattern, add_curly, expression)

        pc = parse.ParsedCommand(expression, self._labels)
        label = str(self.ui.new_label.text()) or 'new component'
        new_id = core.data.ComponentID(label)
        link = parse.ParsedComponentLink(new_id, pc)
        return link

    @property
    def _number_targets(self):
        """
        How many targets are selected
        """
        return len(self.ui.data_list.selectedItems())

    def _add_link_to_targets(self, link):
        """ Add a link to all the selected data """
        for target in self._selected_data():
            target.add_component_link(link)

    def _add_to_expression(self, item):
        """ Add a component list item to the expression editor """
        addition = '%s ' % item.text()
        expression = self.ui.expression
        expression.insertPlainText(addition)

    @staticmethod
    def create_component(collection):
        """Present user with a dialog to define and add new components.

        Parameters
        ----------
        collection : A `DataCollection` to edit
        """
        # pylint: disable=W0212
        widget = CustomComponentWidget(collection)
        while True:
            widget.ui.show()
            if widget.ui.exec_() == QtGui.QDialog.Accepted:
                if len(str(widget.ui.expression.toPlainText())) == 0:
                    QtGui.QMessageBox.critical(widget.ui, "Error", "No expression set",
                                         buttons=QtGui.QMessageBox.Ok)
                elif widget._number_targets == 0:
                    QtGui.QMessageBox.critical(widget.ui, "Error", "Please specify the target dataset(s)",
                                         buttons=QtGui.QMessageBox.Ok)
                elif len(widget.ui.new_label.text()) == 0:
                    QtGui.QMessageBox.critical(widget.ui, "Error", "Please specify the new component name",
                                         buttons=QtGui.QMessageBox.Ok)
                else:
                    link = widget._create_link()
                    if link:
                        widget._add_link_to_targets(link)
                    break
            else:
                break

def main():
    from glue.core.data import Data
    from glue.core.data_collection import DataCollection
    import numpy as np

    x = np.random.random((5, 5))
    y = x * 3
    data = DataCollection(Data(label='test', x=x, y=y))

    CustomComponentWidget.create_component(data)
    for d in data:
        print(d.label)
        for c in d.components:
            print('\t%s' % c)

if __name__ == "__main__":
    main()
