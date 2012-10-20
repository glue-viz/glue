"""
This module provides two classes for managing LayerArtists with Qt.

The LayerArtistModel implements a QtModel to interface with a list of
LayerManagers.

The LayerArtistView is a list widget that displays
these layers, and provides GUI access to the model
"""
#pylint: disable=I0011, W0613, R0913, R0904, W0611
from PyQt4.QtGui import (QColor,
                         QListView, QAbstractItemView, QAction,
                         QPalette)
from PyQt4.QtCore import Qt, QAbstractListModel, QModelIndex, QSize, QTimer

from .qtutil import (PyMimeData, edit_layer_color,
                     edit_layer_symbol, edit_layer_point_size,
                     layer_artist_icon)
from ..clients.layer_artist import LayerArtist, LayerArtistContainer
from . import glue_qt_resources


class LayerArtistModel(QAbstractListModel):
    """A Qt model to manage a list of LayerArtists. Multiple
    views into this model should stay in sync, thanks to Qt.

    To properly maintain sync, any client that uses
    this list of LayerArtists should always edit the
    list in-place (so that the list managed by this model
    and the client are the same object)
    """
    def __init__(self, artists, parent=None):
        super(LayerArtistModel, self).__init__(parent)
        self.artists = artists

    def rowCount(self, parent=None):
        """Number of rows"""
        return len(self.artists)

    def headerData(self, section, orientation, role):
        """Column labels"""
        if role != Qt.DisplayRole:
            return None
        return "%i" % section

    def data(self, index, role):
        """Retrieve data at each index"""
        if not index.isValid():
            return None
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.row_label(index.row())
        if role == Qt.DecorationRole:
            art = self.artists[index.row()]
            result = layer_artist_icon(art)
            return result
        if role == Qt.CheckStateRole:
            art = self.artists[index.row()]
            result = Qt.Checked if art.visible else Qt.Unchecked
            return result

    def flags(self, index):
        result = super(LayerArtistModel, self).flags(index)
        if index.isValid():
            result = (result | Qt.ItemIsEditable | Qt.ItemIsDragEnabled |
                      Qt.ItemIsDropEnabled | Qt.ItemIsUserCheckable)
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

    def removeRow(self, row, parent=None):
        if row < 0 or row >= len(self.artists):
            return False

        self.beginRemoveRows(QModelIndex(), row, row)
        art = self.artists.pop(row)
        art.clear()
        art.redraw()
        self.endRemoveRows()
        return True

    def mimeTypes(self):
        return [PyMimeData.MIME_TYPE]

    def mimeData(self, indexes):
        arts = [self.artists[index.row()] for index in indexes]
        if len(indexes) == 0:
            return 0
        return PyMimeData(arts)

    def supportedDropActions(self):
        return Qt.CopyAction | Qt.MoveAction

    def dropMimeData(self, data, action, row, column, index):
        data = data.data(PyMimeData.MIME_TYPE)
        #list of a single artist. Move
        if isinstance(data, list) and len(data) == 1 and \
                isinstance(data[0], LayerArtist) and data[0] in self.artists:
            self.move_artist(data[0], index.row())
            return True

        return False

    def move_artist(self, artist, row):
        """Move a artist to a different location, and update the zorder"""
        try:
            loc = self.artists.index(artist)
        except ValueError:
            return

        dest = row
        if (loc <= row):
            dest += 1
        if not self.beginMoveRows(QModelIndex(), loc, loc,
                                  QModelIndex(), dest):
            return

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
        return self.artists[row].layer.label

    def change_label(self, row, label):
        """ Reassign the labeel for whatever layer the artist manages"""
        try:
            art = self.artists[row]
            art.layer.label = label
        except IndexError:
            pass

    def add_artist(self, row, artist):
        """Add a new artist"""
        self.beginInsertRows(QModelIndex(), row, row)
        self.artists.insert(row, artist)
        self.endInsertRows()
        self.rowsInserted.emit(self.index(row), row, row)

    def edit_size(self, row):
        index = self.index(row)
        if not index.isValid():
            return
        artist = self.artists[row]
        edit_layer_point_size(artist.layer)
        index = self.index(row)
        self.dataChanged.emit(index, index)

    def edit_color(self, row):
        index = self.index(row)
        if not index.isValid():
            return
        artist = self.artists[row]
        edit_layer_color(artist.layer)
        self.dataChanged.emit(index, index)

    def edit_symbol(self, row):
        index = self.index(row)
        if not index.isValid():
            return
        artist = self.artists[row]
        edit_layer_symbol(artist.layer)
        self.dataChanged.emit(index, index)


class LayerArtistView(QListView):
    """A list view into an artist model. The zorder
    of each artist can be shuffled by dragging and dropping
    items. Right-clicking brings up a menu to edit color, size, and
    marker"""
    def __init__(self, parent=None):
        super(LayerArtistView, self).__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setIconSize(QSize(15, 15))
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        self._set_palette()
        self._create_actions()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.viewport().update)
        self._timer.start(1000)

    def current_artist(self):
        rows = self.selectionModel().selectedRows()
        if len(rows) != 1:
            return
        return self.artists[rows[0].row()]

    def single_selection(self):
        return self.current_artist() is not None

    def current_row(self):
        rows = self.selectionModel().selectedRows()
        if len(rows) != 1:
            return
        return rows[0].row()

    def _set_palette(self):
        p = self.palette()
        c = QColor(240, 240, 240)
        p.setColor(QPalette.Highlight, c)
        p.setColor(QPalette.HighlightedText, QColor(Qt.black))
        self.setPalette(p)

    def _create_actions(self):
        act = QAction('size', self)
        act.triggered.connect(
            lambda: self.model().edit_size(self.current_row()))
        self.addAction(act)

        act = QAction('color', self)
        act.triggered.connect(
            lambda: self.model().edit_color(self.current_row()))
        self.addAction(act)

        act = QAction('symbol', self)
        act.triggered.connect(
            lambda: self.model().edit_symbol(self.current_row()))
        self.addAction(act)

        act = QAction('remove', self)
        act.triggered.connect(
            lambda: self.model().removeRow(self.current_row()))
        self.addAction(act)


class QtLayerArtistContainer(LayerArtistContainer):
    """A subclass of LayerArtistContainer that dispatches to a
    LayerArtistModel"""
    def __init__(self):
        super(QtLayerArtistContainer, self).__init__()
        self.model = LayerArtistModel(self.artists)

    def append(self, artist):
        self._check_duplicate(artist)
        self.model.add_artist(0, artist)
        artist.zorder = max(a.zorder for a in self.artists) + 1
        assert self.artists[0] is artist

    def remove(self, artist):
        try:
            index = self.artists.index(artist)
        except ValueError:
            return
        self.model.removeRow(index)
        assert artist not in self.artists

    def __nonzero__(self):
        return True
