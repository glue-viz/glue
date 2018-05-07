"""
This module provides two classes for managing LayerArtists with Qt.

The LayerArtistModel implements a QtModel to interface with a list of
LayerManagers.

The LayerArtistView is a list widget that displays
these layers, and provides GUI access to the model
"""
# pylint: disable=I0011, W0613, R0913, R0904, W0611

from __future__ import absolute_import, division, print_function

import textwrap
from weakref import WeakKeyDictionary

from qtpy.QtCore import Qt
from qtpy import QtCore, QtWidgets, QtGui
from glue.core.layer_artist import LayerArtistBase, LayerArtistContainer
from glue.core.qt.style_dialog import StyleDialog
from glue.icons.qt import layer_artist_icon
from glue.core.qt.mime import LAYERS_MIME_TYPE
from glue.utils import nonpartial
from glue.utils.qt import PythonListModel, PyMimeData
from glue.core.hub import HubListener
from glue.core.message import LayerArtistEnabledMessage, LayerArtistUpdatedMessage, LayerArtistDisabledMessage


class LayerArtistModel(PythonListModel):

    """A Qt model to manage a list of LayerArtists. Multiple
    views into this model should stay in sync, thanks to Qt.

    To properly maintain sync, any client that uses
    this list of LayerArtists should always edit the
    list in-place (so that the list managed by this model
    and the client are the same object)
    """

    def __init__(self, artists, parent=None):
        super(LayerArtistModel, self).__init__(artists, parent)
        self.artists = artists

    def data(self, index, role):
        """Retrieve data at each index"""
        if not index.isValid() or index.row() >= len(self.artists):
            return None
        if role == Qt.DecorationRole:
            art = self.artists[index.row()]
            result = layer_artist_icon(art)
            return result
        if role == Qt.CheckStateRole:
            art = self.artists[index.row()]
            result = Qt.Checked if art.visible and art.enabled else Qt.Unchecked
            return result
        if role == Qt.ToolTipRole:
            art = self.artists[index.row()]
            if not art.enabled:
                wrapped = textwrap.fill(art.disabled_message, break_long_words=False)
                return wrapped

        return super(LayerArtistModel, self).data(index, role)

    def flags(self, index):
        result = super(LayerArtistModel, self).flags(index)
        if index.isValid() and index.row() < len(self.artists):
            art = self.artists[index.row()]
            if art.enabled:
                result = (result | Qt.ItemIsEditable | Qt.ItemIsDragEnabled |
                          Qt.ItemIsUserCheckable)
            else:
                result = (result & Qt.ItemIsUserCheckable) ^ result
        else:  # only drop between rows, where index isn't valid
            result = result | Qt.ItemIsDropEnabled

        return result

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role == Qt.EditRole:
            self.change_label(index.row(), str(value))
        if role == Qt.CheckStateRole:
            vis = value == Qt.Checked
            self.artists[index.row()].visible = vis
            self.artists[index.row()].redraw()

        self.dataChanged.emit(index, index)
        return True

    def _remove_row(self, row):
        art = self.artists.pop(row)
        art.remove()
        art.redraw()

    def mimeTypes(self):
        return [PyMimeData.MIME_TYPE, LAYERS_MIME_TYPE]

    def mimeData(self, indexes):
        arts = [self.artists[index.row()] for index in indexes]
        layers = [a.layer for a in arts]

        if len(indexes) == 0:
            return 0
        return PyMimeData(arts, **{LAYERS_MIME_TYPE: layers})

    def supportedDropActions(self):
        return Qt.MoveAction

    def dropMimeData(self, data, action, row, column, index):
        data = data.data(PyMimeData.MIME_TYPE)
        # list of a single artist. Move
        if isinstance(data, list) and len(data) == 1 and \
                isinstance(data[0], LayerArtistBase) and \
                data[0] in self.artists:
            self.move_artist(data[0], row)
            return True

        return False

    def move_artist(self, artist, row):
        """Move an artist before the entry in row

        Row could be the end of the list (-> put it at the end)
        """
        if len(self.artists) < 2:  # can't rearrange lenght 0 or 1 list
            return

        try:
            loc = self.artists.index(artist)
        except ValueError:
            return

        dest = row
        if not self.beginMoveRows(QtCore.QModelIndex(), loc, loc,
                                  QtCore.QModelIndex(), dest):
            return
        if dest >= loc:
            row -= 1
        self.artists.pop(loc)
        self.artists.insert(row, artist)
        self._update_zorder()
        self.endMoveRows()

    def _update_zorder(self):
        """Redistribute zorders to match location in the list"""
        zs = [m.zorder for m in self.artists]
        zs = reversed(sorted(zs))
        for z, m in zip(zs, self.artists):
            m.zorder = z
        if len(self.artists) > 0:
            self.artists[0].redraw()

    def row_label(self, row):
        """ The textual label for the row"""
        artist = self.artists[row]
        if hasattr(artist, 'label'):
            return artist.label
        layer = artist.layer
        if hasattr(layer, 'verbose_label'):
            return layer.verbose_label
        return layer.label

    def change_label(self, row, label):
        """ Reassign the labeel for whatever layer the artist manages"""
        try:
            art = self.artists[row]
            art.layer.label = label
        except IndexError:
            pass

    def add_artist(self, row, artist):
        """Add a new artist"""
        self.beginInsertRows(QtCore.QModelIndex(), row, row)
        self.artists.insert(row, artist)
        self.endInsertRows()
        self.rowsInserted.emit(self.index(row), row, row)

    def row_artist(self, row):
        return self.artists[row]

    def artist_row(self, artist):
        return self.artists.index(artist)


class LayerArtistView(QtWidgets.QListView, HubListener):

    """A list view into an artist model. The zorder
    of each artist can be shuffled by dragging and dropping
    items. Right-clicking brings up a menu to edit style or delete"""

    def __init__(self, parent=None, hub=None):
        super(LayerArtistView, self).__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setIconSize(QtCore.QSize(15, 15))
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.setEditTriggers(self.NoEditTriggers)

        self.setMinimumSize(200, 100)
        self._actions = {}
        self._create_actions()

        # Update when any message is emitted which would indicate a change in
        # data collection content, colors, labels, etc. It's easier to simply
        # listen to all events since the viewport update is fast.
        self.hub = hub
        self.hub.subscribe(self, LayerArtistUpdatedMessage, self._update_viewport)
        self.hub.subscribe(self, LayerArtistEnabledMessage, self._layer_enabled_or_disabled)
        self.hub.subscribe(self, LayerArtistDisabledMessage, self._layer_enabled_or_disabled)

    def _update_viewport(self, *args):
        # This forces the widget containing the list view to update/redraw,
        # reflecting any changes in color/labels/content
        self.viewport().update()

    def _layer_enabled_or_disabled(self, *args):

        # This forces the widget containing the list view to update/redraw,
        # reflecting any changes in disabled/enabled layers. If a layer is
        # disabled, it will become unselected.
        self.viewport().update()

        # If a layer artist becomes unselected as a result of the update above,
        # a selection change event is not emitted for some reason, so we force
        # a manual update. If a layer artist was deselected, current_artist()
        # will be None and the options will be hidden for that layer.
        parent = self.parent()
        if parent is not None:
            parent.on_selection_change(self.current_artist())

    def rowsInserted(self, index, start, end):
        super(LayerArtistView, self).rowsInserted(index, start, end)
        # If no rows are currently selected, make sure we select one. We do
        # this to make sure the layer style editor is visible to users straight
        # away.
        parent = self.parent()
        if parent is not None:
            parent.on_artist_add(self.model().artists)
        if self.current_row() is None:
            self.setCurrentIndex(self.model().index(0))

    def selectionChanged(self, selected, deselected):
        super(LayerArtistView, self).selectionChanged(selected, deselected)
        parent = self.parent()
        if parent is not None:
            parent.on_selection_change(self.current_artist())

    def current_artist(self):
        model = self.selectionModel()
        if model is None:
            return
        rows = model.selectedRows()
        if len(rows) != 1:
            return
        return self.model().row_artist(rows[0].row())

    def select_artist(self, artist):
        model = self.selectionModel()
        row = self.model().artist_row(artist)
        model.select(model.model().index(row), model.ClearAndSelect)

    def single_selection(self):
        return self.current_artist() is not None

    def current_row(self):
        model = self.selectionModel()
        if model is None:
            return
        rows = model.selectedRows()
        if len(rows) != 1:
            return
        return rows[0].row()

    def _bottom_left_of_current_index(self):
        idx = self.currentIndex()
        if not idx.isValid():
            return
        rect = self.visualRect(idx)
        pos = self.mapToGlobal(rect.bottomLeft())
        pos.setY(pos.y() + 1)
        return pos

    def _edit_style(self):
        pos = self._bottom_left_of_current_index()
        if pos is None:
            return
        item = self.current_artist().layer
        StyleDialog.dropdown_editor(item, pos, edit_label=False)

    def _create_actions(self):
        act = QtWidgets.QAction('Edit style', self)
        act.triggered.connect(nonpartial(self._edit_style))
        self.addAction(act)

        act = QtWidgets.QAction('Remove', self)
        act.setShortcut(QtGui.QKeySequence(Qt.Key_Backspace))
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(
            lambda *args: self.model().removeRow(self.current_row()))
        self.addAction(act)


class LayerArtistWidget(QtWidgets.QWidget):
    """
    A widget that includes a list of artists (LayerArtistView) and the visual
    options for the layer artists.
    """

    def __init__(self, parent=None, layer_style_widget_cls=None, hub=None):

        super(LayerArtistWidget, self).__init__(parent=parent)

        self.hub = None

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.layer_style_widget_cls = layer_style_widget_cls

        self.layer_list = LayerArtistView(parent=self, hub=hub)
        self.layout.addWidget(self.layer_list)

        self.layer_options = QtWidgets.QWidget()
        self.layer_options_layout = QtWidgets.QStackedLayout()
        self.layer_options.setLayout(self.layer_options_layout)

        self.layout.addWidget(self.layer_options)

        self.setLayout(self.layout)

        self.layout_style_widgets = WeakKeyDictionary()

        self.empty = QtWidgets.QWidget()
        self.layer_options_layout.addWidget(self.empty)

        self.disabled_warning = QtWidgets.QLabel()
        self.disabled_warning.setWordWrap(True)
        self.disabled_warning.setAlignment(Qt.AlignJustify)
        self.padded_warning = QtWidgets.QWidget()
        warning_layout = QtWidgets.QVBoxLayout()
        warning_layout.setContentsMargins(20, 20, 20, 20)
        warning_layout.addWidget(self.disabled_warning)
        self.padded_warning.setLayout(warning_layout)

        self.layer_options_layout.addWidget(self.padded_warning)

    def on_artist_add(self, layer_artists):

        if self.layer_style_widget_cls is None:
            return

        for layer_artist in layer_artists:
            if layer_artist not in self.layout_style_widgets:
                if isinstance(self.layer_style_widget_cls, dict):
                    layer_artist_cls = layer_artist.__class__
                    if layer_artist_cls in self.layer_style_widget_cls:
                        layer_style_widget_cls = self.layer_style_widget_cls[layer_artist_cls]
                    else:
                        return
                else:
                    layer_style_widget_cls = self.layer_style_widget_cls
                self.layout_style_widgets[layer_artist] = layer_style_widget_cls(layer_artist)
                self.layer_options_layout.addWidget(self.layout_style_widgets[layer_artist])

    def on_selection_change(self, layer_artist):

        if layer_artist in self.layout_style_widgets:
            if layer_artist.enabled:
                self.layer_options_layout.setCurrentWidget(self.layout_style_widgets[layer_artist])
                self.disabled_warning.setText('')
            else:
                self.disabled_warning.setText(layer_artist.disabled_message)
                self.layer_options_layout.setCurrentWidget(self.padded_warning)
        else:
            self.layer_options_layout.setCurrentWidget(self.empty)
            self.disabled_warning.setText('')


class QtLayerArtistContainer(LayerArtistContainer):

    """A subclass of LayerArtistContainer that dispatches to a
    LayerArtistModel"""

    def __init__(self):
        super(QtLayerArtistContainer, self).__init__()
        self.model = LayerArtistModel(self.artists)
        self.model.rowsInserted.connect(nonpartial(self._notify))
        self.model.rowsRemoved.connect(nonpartial(self._notify))
        self.model.modelReset.connect(nonpartial(self._notify))

    def append(self, artist):
        self.model.add_artist(0, artist)
        artist.zorder = max(a.zorder for a in self.artists) + 1
        assert self.artists[0] is artist
        self._notify()

    def remove(self, artist):
        if artist in self.artists:
            index = self.artists.index(artist)
            self.model.removeRow(index)
            assert artist not in self.artists
            artist.remove()
            self._notify()

    def __nonzero__(self):
        return True

    __bool__ = __nonzero__
