from PyQt4.QtGui import QWidget, QListWidgetItem

from .ui.component_selector import Ui_ComponentSelector


class ComponentSelector(QWidget):
    """ An interface to view the components and data of a DataCollection

    The widget supports dragging, and stores an instance of the
    dragged ComponentID in the application/py_instance mime type

    The currently-selected componentID is also stored in the
    Component attribute

    Usage:

       >>> widget = ComponentSelector()
       >>> widget.setup(data_collection)
    """
    def __init__(self, parent=None):
        super(ComponentSelector, self).__init__(parent)
        self._data = None
        self._ui = Ui_ComponentSelector()
        self._init_widgets()
        self._connect()

    def _init_widgets(self):
        self._ui.setupUi(self)
        self._ui.component_selector.setDragEnabled(True)

    def _connect(self):
        ds = self._ui.data_selector
        ds.currentIndexChanged.connect(self._set_components)

    def set_current_row(self, row):
        self._ui.component_selector.setCurrentRow(row)

    def set_data_row(self, row):
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
        if index < 0:
            return
        data = self._data[index]
        cids = data.components

        c_list = self._ui.component_selector
        c_list.clear()
        for c in cids:
            item = QListWidgetItem(c.label)
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


def main():  # pragma: no cover
    import glue
    import numpy as np
    from PyQt4.QtGui import QApplication

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

    app = QApplication([''])
    w = ComponentSelector()
    w.setup(dc)
    w.show()
    app.exec_()

if __name__ == "__main__":  # pragma: no cover
    main()
