from __future__ import absolute_import, division, print_function

import os

from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from glue.utils.qt import load_ui
from glue.utils import nonpartial


class ComponentSelector(QtWidgets.QWidget):
    """ An interface to view the components and data of a DataCollection

    Components can be draged and dropped.

    The currently-selected componentID is stored in the
    Component property. The currently-selected Data is stored in the
    Data property.

    Usage:

       >>> widget = ComponentSelector()
       >>> widget.setup(data_collection)
    """

    data_changed = QtCore.Signal()
    component_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super(ComponentSelector, self).__init__(parent)
        self._data = None
        self._ui = load_ui('component_selector.ui', self,
                           directory=os.path.dirname(__file__))
        self._init_widgets()
        self._connect()

    def _init_widgets(self):
        self._ui.component_selector.setDragEnabled(True)
        self._ui.setMinimumWidth(300)

    def _connect(self):
        # attach Qt signals
        ds = self._ui.data_selector
        ds.currentIndexChanged.connect(nonpartial(self._set_components))
        self._ui.component_selector.currentItemChanged.connect(
            lambda *args: self.component_changed.emit())
        self._ui.data_selector.currentIndexChanged.connect(
            lambda *args: self.data_changed.emit())

    def set_current_row(self, row):
        """Select which component is selected

        :param row: Row number
        """
        self._ui.component_selector.setCurrentRow(row)

    def set_data_row(self, row):
        """Select which data object is selected

        :param row: Row number
        """
        self._ui.data_selector.setCurrentIndex(row)

    def setup(self, data_collection):
        """ Set up the widgets.

        :param data_collection: Object to browse
        :type data_colleciton:
           :class:`~glue.core.data_collection.DataCollection`
        """
        self._data = data_collection
        self._set_data()
        self._set_components()

    def _set_components(self):
        """ Set list of component widgets to match current data set """
        index = self._ui.data_selector.currentIndex()
        c_list = self._ui.component_selector
        c_list.clear()
        if index < 0:
            return
        data = self._data[index]

        # Coordinate components
        if len(data.coordinate_components) > 0:
            item = QtWidgets.QListWidgetItem('Coordinate components')
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            c_list.addItem(item)
            for c in data.coordinate_components:
                item = QtWidgets.QListWidgetItem(c.label)
                c_list.addItem(item)
                c_list.set_data(item, c)

        if len(set(data.primary_components) - set(data.coordinate_components)) > 0:
            item = QtWidgets.QListWidgetItem('Main components')
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            c_list.addItem(item)
            for c in data.primary_components:
                if c not in data.coordinate_components:
                    item = QtWidgets.QListWidgetItem(c.label)
                    c_list.addItem(item)
                    c_list.set_data(item, c)

        if len(set(data.derived_components) & set(data.visible_components)) > 0:
            item = QtWidgets.QListWidgetItem('Derived components')
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            c_list.addItem(item)
            for c in data.derived_components:
                item = QtWidgets.QListWidgetItem(c.label)
                c_list.addItem(item)
                c_list.set_data(item, c)

    def _set_data(self):
        """ Populate the data list with data sets in the collection """
        d_list = self._ui.data_selector
        for d in self._data:
            d_list.addItem(d.label)

    @property
    def component(self):
        """Returns the currently-selected ComponentID
        :rtype: :class:`~glue.core.data.ComponentID`
        """
        item = self._ui.component_selector.currentItem()
        return self._ui.component_selector.get_data(item)

    @component.setter
    def component(self, component):
        w = self._ui.component_selector
        for i in range(w.count()):
            item = w.item(i)
            if w.get_data(item) is component:
                w.setCurrentRow(i)
                return
        else:
            raise ValueError("Component not found: %s" % component)

    @property
    def data(self):
        index = self._ui.data_selector.currentIndex()
        if index < 0:
            return
        return self._data[index]

    @data.setter
    def data(self, value):
        if value is None:
            self._ui.data_selector.setCurrentIndex(-1)
            return
        for i, d in enumerate(self._data):
            if d is value:
                self._ui.data_selector.setCurrentIndex(i)
                return
        else:
            raise ValueError("Data is not part of the DataCollection")


def main():  # pragma: no cover
    import glue
    import numpy as np
    from glue.utils.qt import get_qapp

    d = glue.core.Data(label="hi")
    d2 = glue.core.Data(label="there")

    c1 = glue.core.Component(np.array([1, 2, 3]))
    c2 = glue.core.Component(np.array([1, 2, 3]))
    c3 = glue.core.Component(np.array([1, 2, 3]))

    dc = glue.core.DataCollection()
    dc.append(d)
    dc.append(d2)
    d.add_component(c1, "a")
    d.add_component(c2, "b")
    d2.add_component(c3, "c")

    app = get_qapp()
    w = ComponentSelector()
    w.setup(dc)
    w.show()
    app.exec_()


if __name__ == "__main__":  # pragma: no cover
    main()
