from __future__ import absolute_import, division, print_function

from .data_viewer import DataViewer

from ...external.qt.QtGui import QTableView, QAbstractItemView, QItemSelectionModel, QItemSelection
from ...external.qt.QtCore import Qt, QAbstractTableModel, QAbstractItemModel, SIGNAL, QObject

from ...core import message as msg
from ...core.edit_subset_mode import EditSubsetMode
from ...core.subset import ElementSubsetState

from ..qtutil import load_ui

import numpy as np


class DataTableModel(QAbstractTableModel):

    def __init__(self, data):
        super(DataTableModel, self).__init__()
        self._data = data
        self.show_hidden = False

    @property
    def columns(self):
        if self.show_hidden:
            return self._data.components
        else:
            return self._data.visible_components

    def columnCount(self, index=None):
        return len(self.columns)

    def rowCount(self, index=None):
        #Qt bug: Crashes on tables bigger than this
        return min(self._data.size, 71582788)

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.columns[section].label
        elif orientation == Qt.Vertical:
            return str(section)

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            c = self.columns[index.column()]
            idx = np.unravel_index([index.row()], self._data.shape)
            comp = self._data.get_component(c)
            if comp.categorical:
                comp = comp.labels
            else:
                comp = comp.data
            return str(comp[idx][0])


class TableWidget(DataViewer):

    LABEL = "Table Viewer"

    def __init__(self, session, parent=None, widget=None):

        super(TableWidget, self).__init__(session, parent)

        self.ui = load_ui('table_widget')
        self.setCentralWidget(self.ui)

        hdr = self.ui.table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setResizeMode(hdr.Interactive)

        hdr = self.ui.table.verticalHeader()
        hdr.setResizeMode(hdr.Interactive)


        self.ui.table.clicked.connect(self._clicked)

        self.model = None

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
            rows = [x.row() for x in model.selectedRows()]
            subset_state = ElementSubsetState(rows)

            mode = EditSubsetMode()
            mode.update(self.data, subset_state, focus_data=self.data)

    def _add_subset(self, message):

        self.subset = message.subset
        self.ui.table.clearSelection()
        selection_mode = self.ui.table.selectionMode()
        self.ui.table.setSelectionMode(QAbstractItemView.MultiSelection);

        # The following is more efficient than just calling selectRow
        model = self.ui.table.selectionModel()
        for index in message.subset.to_index_list():
            model_index = self.model.createIndex(index, 0)
            model.select(model_index,
                         QItemSelectionModel.Select | QItemSelectionModel.Rows)

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
        self.ui.table.setModel(self.model)
        self.setUpdatesEnabled(True)

    def closeEvent(self, event):
        """
        On close, QT seems to scan through the entire model
        if the data set is big. To sidestep that,
        we swap out with a tiny data set before closing
        """
        from ...core import Data
        d = Data(x=[0])
        self.ui.table.setModel(DataTableModel(d))
        event.accept()
