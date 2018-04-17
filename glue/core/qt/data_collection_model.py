# pylint: disable=E1101,F0401

from __future__ import absolute_import, division, print_function

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from glue.core.hub import HubListener
from glue.core import message as m
from glue.core.decorators import memoize
from glue import core
from glue.core.qt.mime import LAYERS_MIME_TYPE
from glue.icons.qt import layer_icon

from glue.core.qt.style_dialog import StyleDialog
from glue.utils.qt import PyMimeData
from glue.core.message import Message

DATA_IDX = 0
SUBSET_IDX = 1


def full_edit_factory(item, pos):
    StyleDialog.dropdown_editor(item, pos)


def restricted_edit_factory(item, pos):
    StyleDialog.dropdown_editor(item, pos, edit_label=False)


class Item(object):
    edit_factory = None
    glue_data = None
    flags = Qt.ItemIsEnabled
    tooltip = None

    def font(self):
        return QtGui.QFont()

    def icon(self):
        return None

    @property
    def label(self):
        return self._label


class DataCollectionItem(Item):

    def __init__(self, dc):
        self.dc = dc
        self.row = 0
        self.column = 0
        self.parent = None
        self._label = ''
        self.children_count = 2

    @memoize
    def child(self, row):
        if row == DATA_IDX:
            return DataListItem(self.dc, self)
        if row == SUBSET_IDX:
            return SubsetListItem(self.dc, self)
        return None


class DataListItem(Item):

    def __init__(self, dc, parent):
        self.dc = dc
        self.parent = parent
        self.row = DATA_IDX
        self.column = 0
        self._label = 'Data'

    @memoize
    def child(self, row):
        if row < len(self.dc):
            return DataItem(self.dc, row, self)

    @property
    def children_count(self):
        return len(self.dc)

    def font(self):
        result = QtGui.QFont()
        result.setBold(True)
        return result


class DataItem(Item):
    edit_factory = full_edit_factory
    flags = (Qt.ItemIsSelectable | Qt.ItemIsEnabled |
             Qt.ItemIsDragEnabled)

    def __init__(self, dc, row, parent):
        self.dc = dc
        self.row = row
        self.parent = parent
        self.column = 0
        self.children_count = 0

    @property
    def data(self):
        return self.dc[self.row]

    @property
    def glue_data(self):
        return self.data

    @property
    def label(self):
        return self.data.label

    @label.setter
    def label(self, value):
        self.data.label = value

    @property
    def tooltip(self):
        # Return the label as the tooltip - this is useful if filenames are
        # really long and don't fit in the window.
        return self.label

    @property
    def style(self):
        return self.data.style

    def icon(self):
        return layer_icon(self.data)


class SubsetListItem(Item):

    def __init__(self, dc, parent):
        self.dc = dc
        self.parent = parent
        self.row = SUBSET_IDX
        self._label = 'Subsets'
        self.column = 0

    @memoize
    def child(self, row):
        if row < len(self.dc.subset_groups):
            return SubsetGroupItem(self.dc, row, self)

    @property
    def children_count(self):
        return len(self.dc.subset_groups)

    def font(self):
        result = QtGui.QFont()
        result.setBold(True)
        return result


class SubsetGroupItem(Item):
    edit_factory = full_edit_factory
    flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def __init__(self, dc, row, parent):
        self.parent = parent
        self.dc = dc
        self.row = row
        self.column = 0

    @property
    def subset_group(self):
        return self.dc.subset_groups[self.row]

    @property
    def glue_data(self):
        return self.subset_group

    @property
    def label(self):
        return self.subset_group.label

    @label.setter
    def label(self, value):
        self.subset_group.label = value

    @property
    def tooltip(self):
        if type(self.subset_group.subset_state) == core.subset.SubsetState:
            return "Empty subset"

        atts = self.subset_group.subset_state.attributes
        atts = [a for a in atts if isinstance(a, core.ComponentID)]

        if len(atts) > 0:
            lbl = ', '.join(a.label for a in atts)
            return "Selection on %s" % lbl

    @property
    def style(self):
        return self.subset_group.style

    @property
    def children_count(self):
        return len(self.subset_group.subsets)

    @memoize
    def child(self, row):
        return SubsetItem(self.dc, self.subset_group, row, self)

    def icon(self):
        return layer_icon(self.subset_group)


class SubsetItem(Item):
    edit_factory = restricted_edit_factory
    flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled

    def __init__(self, dc, subset_group, subset_idx, parent):
        self.parent = parent
        self.subset_group = subset_group
        self.row = subset_idx
        self.parent = parent
        self.children_count = 0
        self.column = 0

    @property
    def subset(self):
        return self.subset_group.subsets[self.row]

    @property
    def label(self):
        return self.subset.verbose_label

    def icon(self):
        return layer_icon(self.subset)

    @property
    def style(self):
        return self.subset.style

    @property
    def glue_data(self):
        return self.subset


class DataCollectionModel(QtCore.QAbstractItemModel, HubListener):
    new_item = QtCore.Signal(QtCore.QModelIndex)

    def __init__(self, data_collection, parent=None):
        QtCore.QAbstractItemModel.__init__(self, parent)
        HubListener.__init__(self)

        self.data_collection = data_collection
        self.root = DataCollectionItem(data_collection)
        self._items = {}   # map hashes of Model pointers to model items
        # without this reference, PySide clobbers instance
        # data of model items
        self.register_to_hub(self.data_collection.hub)

    def supportedDragActions(self):
        return Qt.CopyAction

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if column != 0:
            return QtCore.QModelIndex()

        if not parent.isValid():
            parent_item = self.root
        else:
            parent_item = self._get_item(parent)
            if parent_item is None:
                return QtCore.QModelIndex()

        child_item = parent_item.child(row)
        if child_item:
            return self._make_index(row, column, child_item)
        else:
            return QtCore.QModelIndex()

    def _get_item(self, index):
        if not index.isValid():
            return None
        return self._items.get(id(index.internalPointer()), None)

    def _make_index(self, row, column, item):
        if item is not None:
            result = self.createIndex(row, column, item)
            self._items[id(result.internalPointer())] = item
            assert result.internalPointer() is item
            return result
        return self.createIndex(row, column)

    def to_indices(self, items):
        """Translate a list of Data, Subset, or SubsetGroups
        to a list of indices"""
        result = []
        for item in items:
            if isinstance(item, core.Data):
                idx = self.data_index(list(self.data_collection).index(item))
            elif isinstance(item, core.SubsetGroup):
                idx = self.subsets_index(
                    self.data_collection.subset_groups.index(item))
            elif isinstance(item, core.subset_group.GroupedSubset):
                grp = item.group
                idx = self.subsets_index(
                    self.data_collection.subset_groups.index(grp))
                row = list(self.data_collection).index(item.data)
                idx = self.index(row, idx)
            else:
                raise NotImplementedError(type(item))
            result.append(idx)
        return result

    def flags(self, index=QtCore.QModelIndex()):
        item = self._get_item(index)
        if item is None:
            return Qt.NoItemFlags
        else:
            return item.flags

    def data(self, index, role):
        if not index.isValid():
            return

        dispatch = {
            Qt.DisplayRole: self._display_data,
            Qt.FontRole: self._font_data,
            Qt.DecorationRole: self._icon_data,
            Qt.UserRole: self._user_data,
            Qt.ToolTipRole: self._tooltip_data}

        if role in dispatch:
            return dispatch[role](index)

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False
        try:
            self._get_item(index).label = value
            return True
        except AttributeError:
            return False

    def _tooltip_data(self, index):
        tooltip = self._get_item(index).tooltip
        return tooltip

    def _user_data(self, index):
        return self._get_item(index)

    def _display_data(self, index):
        return self._get_item(index).label

    def _font_data(self, index):
        item = self._get_item(index)
        return item.font()

    def _icon_data(self, index):
        return self._get_item(index).icon()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        return ''

    def data_index(self, data_number=None):
        """
        Fetch the QtCore.QModelIndex for a given data index,
        or the index for the parent data item

        :param data_number: position of data set to fetch, or None
        """
        base = self.index(DATA_IDX, 0)
        if data_number is None:
            return base
        return self.index(data_number, 0, base)

    def subsets_index(self, subset_number=None):
        """
        Fetch the QtCore.QModelIndex for a given subset,
        or the index for the parent subset item

        :param data_number: position of subset group to fetch, or None
        """
        base = self.index(SUBSET_IDX, 0)
        assert isinstance(self._get_item(base), SubsetListItem)
        if subset_number is None:
            return base
        return self.index(subset_number, 0, base)

    def rowCount(self, index=QtCore.QModelIndex()):
        item = self._get_item(index)

        if item is None:
            return self.root.children_count

        return item.children_count

    def parent(self, index=None):

        if index is None:  # overloaded QtCore.QObject.parent()
            return QtCore.QObject.parent(self)

        item = self._get_item(index)
        if item is None:
            return QtCore.QModelIndex()

        return self._make_index(item.row, item.column, item.parent)

    def columnCount(self, index):
        return 1

    def register_to_hub(self, hub):
        for msg in [m.DataCollectionDeleteMessage,
                    m.SubsetDeleteMessage]:
            hub.subscribe(self, msg, self.invalidate)

        hub.subscribe(self, m.DataCollectionAddMessage, self._on_add_data)
        hub.subscribe(self, m.SubsetCreateMessage, self._on_add_subset)

    def _on_add_data(self, message):
        self.invalidate()
        idx = self.data_index(len(self.data_collection) - 1)
        self.new_item.emit(idx)

    def _on_add_subset(self, message):
        self.invalidate()
        idx = self.subsets_index(len(self.data_collection.subset_groups) - 1)
        self.new_item.emit(idx)

    def invalidate(self, *args):
        self.root = DataCollectionItem(self.data_collection)
        self._items.clear()
        self.layoutChanged.emit()

    def glue_data(self, indices):
        """ Given a list of indices, return a list of all selected
        Data, Subset, and SubsetGroup objects.
        """
        items = [self._get_item(idx) for idx in indices]
        items = [item.glue_data for item in items]
        return items

    def mimeData(self, indices):
        data = self.glue_data(indices)
        result = PyMimeData(data, **{LAYERS_MIME_TYPE: data})
        self._mime = result  # hold reference to prevent segfault
        return result

    def mimeTypes(self):
        return [LAYERS_MIME_TYPE]


class DataCollectionView(QtWidgets.QTreeView, HubListener):
    selection_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super(DataCollectionView, self).__init__(parent)
        self.doubleClicked.connect(self._edit)

        # only edit label on model.new_item
        self.setItemDelegate(LabeledDelegate())
        self.setEditTriggers(self.NoEditTriggers)

        self.setIconSize(QtCore.QSize(16, 16))

    def selected_layers(self):
        idxs = self.selectedIndexes()
        return self._model.glue_data(idxs)

    def set_selected_layers(self, layers):
        sm = self.selectionModel()
        idxs = self._model.to_indices(layers)
        self.select_indices(*idxs)

    def select_indices(self, *indices):
        sm = self.selectionModel()
        sm.clearSelection()
        for idx in indices:
            sm.select(idx, sm.Select)

    def set_data_collection(self, data_collection):
        self._model = DataCollectionModel(data_collection)
        self.setModel(self._model)

        sm = QtCore.QItemSelectionModel(self._model)
        sm.selectionChanged.connect(lambda *args:
                                    self.selection_changed.emit())
        self.setSelectionModel(sm)

        self.setRootIsDecorated(False)
        self.setExpandsOnDoubleClick(False)
        self.expandToDepth(0)
        self._model.layoutChanged.connect(lambda: self.expandToDepth(0))
        self._model.layoutChanged.connect(self.selection_changed.emit)
        self._model.new_item.connect(self.select_indices)

        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)

        # Update when any message is emitted which would indicate a change in
        # data collection content, colors, labels, etc. It's easier to simply
        # listen to all events since the viewport update is fast.
        data_collection.hub.subscribe(self, Message, handler=self._update_viewport)

    def _update_viewport(self, *args, **kwargs):
        # This forces the widget containing the list view to update/redraw,
        # reflecting any changes in color/labels/content
        try:
            self.viewport().update()
        except RuntimeError:
            pass

    def edit_label(self, index):
        if not (self._model.flags(index) & Qt.ItemIsEditable):
            return
        self.edit(index)

    def _edit(self, index):
        item = self._model.data(index, role=Qt.UserRole)
        if item is None or item.edit_factory is None:
            return

        rect = self.visualRect(index)
        pos = self.mapToGlobal(rect.bottomLeft())
        pos.setY(pos.y() + 1)
        item.edit_factory(pos)


class LabeledDelegate(QtWidgets.QStyledItemDelegate):

    """ Add placeholder text to default delegate """

    def setEditorData(self, editor, index):
        super(LabeledDelegate, self).setEditorData(editor, index)
        label = index.model().data(index, role=Qt.DisplayRole)
        editor.selectAll()
        editor.setText(label)


if __name__ == "__main__":

    from glue.utils.qt import get_qapp
    from qtpy import QtWidgets
    from glue.core import Data, DataCollection

    app = get_qapp()

    dc = DataCollection()
    dc.append(Data(label='w'))

    view = DataCollectionView()
    view.set_data_collection(dc)
    view.show()
    view.raise_()
    dc.extend([Data(label='x', x=[1, 2, 3]),
               Data(label='y', y=[1, 2, 3]),
               Data(label='z', z=[1, 2, 3])])
    app.exec_()
