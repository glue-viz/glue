from .data_viewer import DataViewer

from ...external.qt.QtGui import QTableView
from ...external.qt.QtCore import Qt, QAbstractTableModel

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
            return str(self._data[c, idx][0])


class TableWidget(DataViewer):
    def __init__(self, dc, parent=None):
        super(TableWidget, self).__init__(dc, parent)
        self.widget = QTableView()
        self.setCentralWidget(self.widget)

        hdr = self.widget.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setResizeMode(hdr.Interactive)

        hdr = self.widget.verticalHeader()
        hdr.setResizeMode(hdr.Interactive)

    def __str__(self):
        return "Table Widget"

    def unregister(self, hub):
        pass

    def add_data(self, data):
        self.set_data(data)
        return True

    def add_subset(self, subset):
        self.set_data(subset.data)
        return True

    def set_data(self, data):
        self.setUpdatesEnabled(False)
        model = DataTableModel(data)
        self.widget.setModel(model)
        self.setUpdatesEnabled(True)

    def closeEvent(self, event):
        """
        On close, QT seems to scan through the entire model
        if the data set is big. To sidestep that,
        we swap out with a tiny data set before closing
        """
        from ...core import Data
        d = Data(x=[0])
        self.widget.setModel(DataTableModel(d))
        event.accept()


def main():
    from glue.core import Data, DataCollection
    from glue.qt import get_qapp
    import numpy as np

    app = get_qapp()

    sz = 10 ** 5
    x = np.random.normal(0, 1, sz)
    y = np.random.random(sz)
    d = Data(x=x, y=y)
    dc = DataCollection([d])

    tw = TableWidget(dc)
    tw.set_data(d)

    tw.show()

    app.exec_()

if __name__ == "__main__":
    main()
