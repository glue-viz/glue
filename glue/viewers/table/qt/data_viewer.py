from __future__ import absolute_import, division, print_function

import os
import numpy as np

from qtpy.QtCore import Qt
from qtpy import QtCore, QtGui, QtWidgets
from matplotlib.colors import ColorConverter

from glue.utils.qt import get_qapp
from glue.config import viewer_tool
from glue.core import BaseData, Data
from glue.utils.qt import load_ui
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.common.qt.toolbar import BasicToolbar
from glue.viewers.common.qt.tool import CheckableTool
from glue.viewers.common.layer_artist import LayerArtist
from glue.core.subset import ElementSubsetState
from glue.utils.colors import alpha_blend_colors
from glue.utils.qt import mpl_to_qt_color, messagebox_on_error
from glue.core.exceptions import IncompatibleAttribute
from glue.viewers.table.compat import update_table_viewer_state

__all__ = ['TableViewer', 'TableLayerArtist']

COLOR_CONVERTER = ColorConverter()


class DataTableModel(QtCore.QAbstractTableModel):

    def __init__(self, table_viewer):
        super(DataTableModel, self).__init__()
        if table_viewer.data.ndim != 1:
            raise ValueError("Can only use Table widget for 1D data")
        self._table_viewer = table_viewer
        self._data = table_viewer.data
        self.show_coords = False
        self.order = np.arange(self._data.shape[0])

    def data_changed(self):
        top_left = self.index(0, 0)
        bottom_right = self.index(self.columnCount(), self.rowCount())
        self.dataChanged.emit(top_left, bottom_right)
        self.layoutChanged.emit()

    @property
    def columns(self):
        if self.show_coords:
            return self._data.components
        else:
            return self._data.main_components + self._data.derived_components

    def columnCount(self, index=None):
        return len(self.columns)

    def rowCount(self, index=None):
        # Qt bug: Crashes on tables bigger than this
        return min(self._data.size, 71582788)

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            column_name = self.columns[section].label
            units = self._data.get_component(self.columns[section]).units
            if units != '':
                column_name += "\n{0}".format(units)
            return column_name
        elif orientation == Qt.Vertical:
            return str(self.order[section])

    def data(self, index, role):

        if not index.isValid():
            return None

        if role == Qt.DisplayRole:

            c = self.columns[index.column()]
            idx = self.order[index.row()]
            comp = self._data[c]
            if isinstance(comp[idx], bytes):
                return comp[idx].decode('ascii')
            else:
                return str(comp[idx])

        elif role == Qt.BackgroundRole:

            idx = self.order[index.row()]

            # Find all subsets that this index is part of
            colors = []
            for layer_artist in self._table_viewer.layers[::-1]:
                if isinstance(layer_artist.layer, BaseData):
                    continue
                if layer_artist.visible:
                    subset = layer_artist.layer
                    try:
                        if subset.to_mask(view=slice(idx, idx + 1))[0]:
                            colors.append(subset.style.color)
                    except IncompatibleAttribute as exc:
                        layer_artist.disable_invalid_attributes(*exc.args)
                    else:
                        layer_artist.enabled = True

            # Blend the colors using alpha blending
            if len(colors) > 0:
                color = alpha_blend_colors(colors, additional_alpha=0.5)
                color = mpl_to_qt_color(color)
                return QtGui.QBrush(color)

    def sort(self, column, ascending):
        c = self.columns[column]
        comp = self._data.get_component(c)
        self.order = np.argsort(comp.data)
        if ascending == Qt.DescendingOrder:
            self.order = self.order[::-1]
        self.layoutChanged.emit()


class TableLayerArtist(LayerArtist):

    def __init__(self, table_viewer, viewer_state, layer_state=None, layer=None):
        self._table_viewer = table_viewer
        super(TableLayerArtist, self).__init__(viewer_state,
                                               layer_state=layer_state,
                                               layer=layer)

    def _refresh(self):
        self._table_viewer.model.data_changed()

    def redraw(self):
        self._refresh()

    def update(self):
        self._refresh()

    def clear(self):
        self._refresh()


@viewer_tool
class RowSelectTool(CheckableTool):

    tool_id = 'table:rowselect'
    icon = 'glue_row_select'
    action_text = 'Select row(s)'
    tool_tip = ('Select rows by clicking on rows and pressing enter '
                'once the selection is ready to be applied')
    status_tip = ('CLICK to select, press ENTER to finalize selection, '
                  'ALT+CLICK or ALT+UP/DOWN to apply selection immediately')

    def __init__(self, viewer):
        super(RowSelectTool, self).__init__(viewer)
        self.deactivate()

    def activate(self):
        self.viewer.ui.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    def deactivate(self):
        # Don't do anything if the viewer has already been closed
        if self.viewer is None:
            return
        self.viewer.ui.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.viewer.ui.table.clearSelection()


class TableViewWithSelectionSignal(QtWidgets.QTableView):

    selection_changed = QtCore.Signal()

    def selectionChanged(self, *args, **kwargs):
        self.selection_changed.emit()
        super(TableViewWithSelectionSignal, self).selectionChanged(*args, **kwargs)


class TableViewer(DataViewer):

    LABEL = "Table Viewer"

    _toolbar_cls = BasicToolbar
    _data_artist_cls = TableLayerArtist
    _subset_artist_cls = TableLayerArtist

    _inherit_tools = True
    tools = ['table:rowselect']

    def __init__(self, session, state=None, parent=None, widget=None):

        super(TableViewer, self).__init__(session, state=state, parent=parent)

        self.ui = load_ui('data_viewer.ui',
                          directory=os.path.dirname(__file__))
        self.setCentralWidget(self.ui)

        hdr = self.ui.table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(hdr.Interactive)

        hdr = self.ui.table.verticalHeader()
        hdr.setSectionResizeMode(hdr.Interactive)

        self.model = None

        self.ui.table.selection_changed.connect(self.selection_changed)

        self.state.add_callback('layers', self._on_layers_changed)
        self._on_layers_changed()

    def selection_changed(self):
        app = get_qapp()
        if app.queryKeyboardModifiers() == Qt.AltModifier:
            self.finalize_selection(clear=False)

    def keyPressEvent(self, event):
        if self.toolbar.active_tool is self.toolbar.tools['table:rowselect']:
            if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
                self.finalize_selection()
        super(TableViewer, self).keyPressEvent(event)

    def finalize_selection(self, clear=True):
        model = self.ui.table.selectionModel()
        selected_rows = [self.model.order[x.row()] for x in model.selectedRows()]
        subset_state = ElementSubsetState(indices=selected_rows, data=self.data)
        mode = self.session.edit_subset_mode
        mode.update(self._data, subset_state, focus_data=self.data)
        if clear:
            # We block the signals here to make sure that we don't update
            # the subset again once the selection is cleared.
            self.ui.table.blockSignals(True)
            self.ui.table.clearSelection()
            self.ui.table.blockSignals(False)

    def _on_layers_changed(self, *args):
        for layer_state in self.state.layers:
            if isinstance(layer_state.layer, BaseData):
                break
        else:
            return
        self.data = layer_state.layer
        self.setUpdatesEnabled(False)
        self.model = DataTableModel(self)
        self.ui.table.setModel(self.model)
        self.setUpdatesEnabled(True)

    @messagebox_on_error("Failed to add data")
    def add_data(self, data):
        with self._layer_artist_container.ignore_empty():
            self.state.layers[:] = []
            return super(TableViewer, self).add_data(data)

    def closeEvent(self, event):
        """
        On close, Qt seems to scan through the entire model
        if the data set is big. To sidestep that,
        we swap out with a tiny data set before closing
        """
        super(TableViewer, self).closeEvent(event)
        d = Data(x=[0])
        self.ui.table.setModel(DataTableModel(d))
        event.accept()

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self, self.state, layer=layer, layer_state=layer_state)

    def layer_view(self):
        return QtWidgets.QWidget()

    @staticmethod
    def update_viewer_state(rec, context):
        return update_table_viewer_state(rec, context)
