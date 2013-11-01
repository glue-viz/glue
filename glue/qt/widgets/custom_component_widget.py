from ...external.qt.QtGui import QDialog

from ... import core
from ...core import parse

from ..qtutil import load_ui



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


class CustomComponentWidget(QDialog):
    """ Dialog to add derived components to data via parsed commands """
    def __init__(self, collection, parent=None):
        super(CustomComponentWidget, self).__init__(parent)
        self._labels = {}
        self._data = {}
        self.ui = load_ui('custom_component_widget', self)

        self._collection = collection
        self._gather_components()
        self._gather_data()
        self._init_widgets()
        self._connect()

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
                label = "%s  (%s)" % (c, data.label)
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
        expression = str(self.ui.expression.text())
        pc = parse.ParsedCommand(expression, self._labels)
        label = str(self.ui.new_label.text()) or 'new component'
        new_id = core.data.ComponentID(label)
        link = parse.ParsedComponentLink(new_id, pc)
        return link

    def _add_link_to_targets(self, link):
        """ Add a link to all the selected data """
        for target in self._selected_data():
            target.add_component_link(link)

    def _add_to_expression(self, item):
        """ Add a component list item to the expression editor """
        addition = ' {%s} ' % item.text()
        expression = self.ui.expression
        pos = expression.cursorPosition()
        text = str(expression.displayText())
        expression.setText(text[:pos] + addition + text[pos:])

    @staticmethod
    def create_component(collection):
        """Present user with a dialog to define and add new components.

        Parameters
        ----------
        collection : A `DataCollection` to edit
        """
        # pylint: disable=W0212
        widget = CustomComponentWidget(collection)
        widget.show()
        if widget.exec_() == QDialog.Accepted:
            link = widget._create_link()
            if link:
                widget._add_link_to_targets(link)


def main():
    import sys
    import glue
    from glue.core.datq import Data
    from glue.core.data_collection import DataCollection
    import numpy as np

    x = np.random.random((5,5))
    data = DataCollection(Data(x=x))

    CustomComponentWidget.create_component(data)
    for d in dc:
        print d.label
        for c in d.components:
            print '\t%s' % c

if __name__ == "__main__":
    main()
