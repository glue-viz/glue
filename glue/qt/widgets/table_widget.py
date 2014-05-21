from __future__ import absolute_import, division, print_function

from .data_viewer import DataViewer

from ...external.qt.QtGui import QTableView, QAction, QKeySequence
from ...external.qt.QtCore import Qt, QAbstractTableModel
from .image_widget import StandalonePlotWindow
from ..qtutil import nonpartial, mpl_to_qt4_color
from ...core import message as msg

import numpy as np


class DataTableModel(QAbstractTableModel):

    def __init__(self, data):
        super(DataTableModel, self).__init__()
        self._data = data
        self.show_hidden = False
        self._popup = None

    @property
    def columns(self):
        if self.show_hidden:
            return self._data.components
        else:
            return self._data.visible_components

    def plottable(self, idx):
        c = self.columns[idx.column()]
        return _dimension(self._data.dtype(c)) != 0

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
            return str(section)

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            c = self.columns[index.column()]
            nd = _dimension(self._data.dtype(c))
            if nd != 0:
                return '%iD array' % nd
            idx = np.unravel_index([index.row()], self._data.shape)
            return str(self._data[c, idx][0])

        if role == Qt.UserRole:
            c = self.columns[index.column()]
            idx = np.unravel_index([index.row()], self._data.shape)
            return self._data[c, idx]

        if role == Qt.BackgroundColorRole:
            c = self.columns[index.column()]
            idx = np.unravel_index([index.row()], self._data.shape)
            for sub in self._data.subsets:
                if sub.to_mask(idx):
                    return mpl_to_qt4_color(sub.style.color, alpha=0.2)


class TableWidget(DataViewer):
    LABEL = "Table"

    def __init__(self, session, parent=None):
        super(TableWidget, self).__init__(session, parent)
        self.widget = QTableView()
        self.setCentralWidget(self.widget)

        hdr = self.widget.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setResizeMode(hdr.Interactive)

        hdr = self.widget.verticalHeader()
        hdr.setResizeMode(hdr.Interactive)
        self._setup_actions()

        self._data = None
        self._popup_plot = None

    def _setup_actions(self):
        act = QAction("Visualize", self)
        act.triggered.connect(nonpartial(self._popup_viewer))
        act.setShortcut(QKeySequence("v"))

        self.addAction(act)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

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
        self._data = data
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

    def _popup_viewer(self):
        idxs = self.widget.selectedIndexes()
        m = self.widget.model()
        idxs = [i for i in idxs if m.plottable(i)]
        if self._popup_plot is None:
            self._popup_plot = StandalonePlotWindow()
            self._popup_plot.mdi_wrap()
            self._session.application.add_widget(self._popup_plot, label='')

        self._popup_plot.clear()
        ax = self._popup_plot.axes

        for i in idxs:
            data = m.data(i, role=Qt.UserRole)
            data = data[data.dtype.names[0]]
            ax.plot(*data, label="Row %i" % i.row())
            ax.legend()

        self._popup_plot.show()
        self._popup_plot.redraw()

    def keyPressEvent(self, event):
        if (event.modifiers() and Qt.ControlModifier) and event.key() == Qt.Key_A:
            self._move_to_left()

        if (event.modifiers() and Qt.ControlModifier) and event.key() == Qt.Key_E:
            self._move_to_right()

    def _move_to_left(self):
        sb = self.widget.horizontalScrollBar()
        sb.setValue(sb.minimum())

    def _move_to_right(self):
        sb = self.widget.horizontalScrollBar()
        sb.setValue(sb.maximum())

    def sync(self):
        self.widget.viewport().update()

    def register_to_hub(self, hub):
        super(TableWidget, self).register_to_hub(hub)
        hub.subscribe(self, msg.SubsetUpdateMessage,
                      nonpartial(self.sync))


def _dimension(dtype):
    if dtype.names is None:
        return 0
    return len(dtype[0].shape)
