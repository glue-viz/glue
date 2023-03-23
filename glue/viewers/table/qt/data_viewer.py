import os
from functools import lru_cache

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
from glue.viewers.common.tool import CheckableTool
from glue.viewers.common.layer_artist import LayerArtist
from glue.core.subset import ElementSubsetState
from glue.utils.colors import alpha_blend_colors
from glue.utils.qt import mpl_to_qt_color, messagebox_on_error
from glue.core.exceptions import IncompatibleAttribute
from glue.viewers.table.compat import update_table_viewer_state

try:
    import dask.array as da
    DASK_INSTALLED = True
except ImportError:
    DASK_INSTALLED = False

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
        self._update_visible()

    def data_changed(self):
        top_left = self.index(0, 0)
        bottom_right = self.index(self.columnCount(), self.rowCount())
        self._update_visible()
        self.data_by_row_and_column.cache_clear()
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
        return min(self.order_visible.size, 71582788)

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            column_name = self.columns[section].label
            units = self._data.get_component(self.columns[section]).units
            if units is not None and units != '':
                column_name += "\n{0}".format(units)
            return column_name
        elif orientation == Qt.Vertical:
            return str(self.order_visible[section])

    def data(self, index, role):

        if not index.isValid():
            return None

        return self.data_by_row_and_column(index.row(), index.column(), role)

    # The data() method gets called many times, often with the same parameters,
    # for example if bringing the window to the foreground/background, shifting
    # up/down/left/right by one cell, etc. This can be very slow when e.g. dask
    # columns are present so we cache the most recent 65536 calls which should
    # have a reasonably sensible memory footprint.
    @lru_cache(maxsize=65536)
    def data_by_row_and_column(self, row, column, role):

        if role == Qt.DisplayRole:

            c = self.columns[column]
            idx = self.order_visible[row]
            comp = self._data[c]
            value = comp[idx]
            if isinstance(value, bytes):
                return value.decode('ascii')
            else:
                if DASK_INSTALLED and isinstance(value, da.Array):
                    return str(value.compute())
                else:
                    return str(comp[idx])

        elif role == Qt.BackgroundRole:

            idx = self.order_visible[row]

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
                        # Only disable the layer if enabled, as otherwise we
                        # will recursively call clear and _refresh, causing
                        # an infinite loop and performance issues.
                        # Also make sure that a disabled layer is not visible
                        if layer_artist.enabled:
                            layer_artist.disable_invalid_attributes(*exc.args)
                            layer_artist.visible = False
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
        self._update_visible()
        self.data_by_row_and_column.cache_clear()
        self.layoutChanged.emit()

    def _update_visible(self):
        """
        Given which layers are visible or not, convert order to order_visible.
        """

        self.data_by_row_and_column.cache_clear()

        # First, if the data layer is visible, show all rows
        for layer_artist in self._table_viewer.layers:
            if layer_artist.visible and isinstance(layer_artist.layer, BaseData):
                self.order_visible = self.order
                return

        # If not then we need to show only the rows with visible subsets
        visible = np.zeros(self.order.shape, dtype=bool)
        for layer_artist in self._table_viewer.layers:
            if layer_artist.visible:
                mask = layer_artist.layer.to_mask()[self.order]
                if DASK_INSTALLED and isinstance(mask, da.Array):
                    mask = mask.compute()
                visible |= mask

        self.order_visible = self.order[visible]


class TableLayerArtist(LayerArtist):

    def __init__(self, table_viewer, viewer_state, layer_state=None, layer=None):
        self._table_viewer = table_viewer
        super(TableLayerArtist, self).__init__(viewer_state,
                                               layer_state=layer_state,
                                               layer=layer)
        self.redraw()

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

    inherit_tools = False
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

        self.data = None
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
        selected_rows = [self.model.order_visible[x.row()] for x in model.selectedRows()]
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

        # If we aren't changing the data layer, we don't need to
        # reset the model, just update visible rows
        if layer_state.layer == self.data:
            self.model._update_visible()
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

    @messagebox_on_error("Failed to add subset")
    def add_subset(self, subset):

        if self.data is None:
            self.add_data(subset.data)
            self.state.layers[0].visible = False
        elif subset.data != self.data:
            raise ValueError("subset parent data does not match existing table data")

        return super(TableViewer, self).add_subset(subset)

    @property
    def window_title(self):
        if len(self.state.layers) > 0:
            return 'Table: ' + self.state.layers[0].layer.label
        else:
            return 'Table'

    def closeEvent(self, event):
        """
        On close, Qt seems to scan through the entire model
        if the data set is big. To sidestep that,
        we swap out with a tiny data set before closing
        """
        super(TableViewer, self).closeEvent(event)
        if self.model is not None:
            self.model._data = Data(x=[0])
        event.accept()

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self, self.state, layer=layer, layer_state=layer_state)

    @staticmethod
    def update_viewer_state(rec, context):
        return update_table_viewer_state(rec, context)
