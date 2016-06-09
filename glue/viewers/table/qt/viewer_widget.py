from __future__ import absolute_import, division, print_function

import os
import numpy as np

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui, QtCore, is_pyqt5
from glue.core.subset import ElementSubsetState
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core import message as msg
from glue.utils.qt import load_ui
from glue.viewers.common.qt.data_viewer import DataViewer


class DataTableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(DataTableModel, self).__init__()
        if data.ndim != 1:
            raise ValueError("Can only use Table widget for one-dimensional data")
        self._data = data
        self.show_hidden = False
        self.order = np.arange(data.shape[0])

    @property
    def columns(self):
        if self.show_hidden:
            return self._data.components
        else:
            return self._data.visible_components

    def columnCount(self, index=None):
        return len(self.columns)

    def rowCount(self, index=None):
        # Qt bug: Crashes on tables bigger than this
        return min(self._data.size, 71582788)

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.columns[section].label
        elif orientation == Qt.Vertical:
            return str(self.order[section])

    def data(self, index, role):

        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            c = self.columns[index.column()]
            idx = self.order[index.row()]
            comp = self._data.get_component(c)
            if comp.categorical:
                comp = comp.labels
            else:
                comp = comp.data
            if isinstance(comp[idx], bytes):
                return comp[idx].decode('ascii')
            else:
                return str(comp[idx])

    def sort(self, column, ascending):
        c = self.columns[column]
        comp = self._data.get_component(c)
        if comp.categorical:
            self.order = np.argsort(comp.labels)
        else:
            self.order = np.argsort(comp.data)
        if ascending == Qt.DescendingOrder:
            self.order = self.order[::-1]
        self.layoutChanged.emit()


class TableWidget(DataViewer):

    LABEL = "Table Viewer"

    def __init__(self, session, parent=None, widget=None):

        super(TableWidget, self).__init__(session, parent)

        self.ui = load_ui('viewer_widget.ui',
                          directory=os.path.dirname(__file__))
        self.setCentralWidget(self.ui)

        hdr = self.ui.table.horizontalHeader()
        hdr.setStretchLastSection(True)

        if is_pyqt5():
            hdr.setSectionResizeMode(hdr.Interactive)
        else:
            hdr.setResizeMode(hdr.Interactive)

        hdr = self.ui.table.verticalHeader()

        if is_pyqt5():
            hdr.setSectionResizeMode(hdr.Interactive)
        else:
            hdr.setResizeMode(hdr.Interactive)

        self.ui.table.clicked.connect(self._clicked)

        self.model = None
        self.subset = None
        self.selected_rows = []

    def register_to_hub(self, hub):

        super(TableWidget, self).register_to_hub(hub)

        dfilter = lambda x: True
        dcfilter = lambda x: True
        subfilter = lambda x: True

        hub.subscribe(self, msg.SubsetCreateMessage,
                      handler=self._add_subset,
                      filter=dfilter)

        hub.subscribe(self, msg.SubsetUpdateMessage,
                      handler=self._update_subset,
                      filter=subfilter)

        hub.subscribe(self, msg.SubsetDeleteMessage,
                      handler=self._remove_subset)

        hub.subscribe(self, msg.DataUpdateMessage,
                      handler=self.update_window_title)

    def _clicked(self, mode):
        self._broadcast_selection()

    def _broadcast_selection(self):

        if self.data is not None:

            model = self.ui.table.selectionModel()
            self.selected_rows = [self.model.order[x.row()] for x in model.selectedRows()]
            subset_state = ElementSubsetState(self.selected_rows)

            mode = EditSubsetMode()
            mode.update(self.data, subset_state, focus_data=self.data)

    def _add_subset(self, message):
        self.subset = message.subset
        self.selected_rows = self.subset.to_index_list()
        self._update_selection()

    def _update_selection(self):

        self.ui.table.clearSelection()
        selection_mode = self.ui.table.selectionMode()
        self.ui.table.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)

        # The following is more efficient than just calling selectRow
        model = self.ui.table.selectionModel()
        for index in self.selected_rows:
            index = self.model.order[index]
            model_index = self.model.createIndex(index, 0)
            model.select(model_index,
                         QtGui.QItemSelectionModel.Select | QtGui.QItemSelectionModel.Rows)

        self.ui.table.setSelectionMode(selection_mode)

    def _update_subset(self, message):
        self._add_subset(message)

    def _remove_subset(self, message):
        self.ui.table.clearSelection()

    def _update_data(self, message):
        self.set_data(message.data)

    def unregister(self, hub):
        pass

    def add_data(self, data):
        self.data = data
        self.set_data(data)
        return True

    def add_subset(self, subset):
        return True

    def set_data(self, data):
        self.setUpdatesEnabled(False)
        self.model = DataTableModel(data)
        self.model.layoutChanged.connect(self._update_selection)
        self.ui.table.setModel(self.model)
        self.setUpdatesEnabled(True)

    def closeEvent(self, event):
        """
        On close, QT seems to scan through the entire model
        if the data set is big. To sidestep that,
        we swap out with a tiny data set before closing
        """
        from glue.core import Data
        d = Data(x=[0])
        self.ui.table.setModel(DataTableModel(d))
        event.accept()
