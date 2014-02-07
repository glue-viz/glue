from ..external.qt.QtCore import (QAbstractItemModel, QModelIndex,
                                  QObject, Qt, QTimer)
from ..external.qt.QtGui import (QFont, QTreeView)

from .qtutil import layer_icon
from ..core import message as m
from ..core.hub import HubListener

DATA_IDX = 0
SUBSET_IDX = 1


def full_edit_factory(item, pos, **kwargs):
    s = StyleDialog(item, **kwargs)
    s.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)

    pos = s.mapFromGlobal(pos)
    s.move(pos)
    if s.exec_() == s.Accepted:
        s.update_style()


def restricted_edit_factory(item, rect):
    return full_edit_factory(item, rect, edit_label=False)


class Item(object):
    edit_factory = None

    def font(self):
        return QFont()

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

    def child(self, row):
        if row == DATA_IDX:
            return DataListItem(self.dc, self)
        if row == SUBSET_IDX:
            return SubsetListItem(self.dc, self)
        return None

    def remove(self):
        return False


class DataListItem(Item):
    def __init__(self, dc, parent):
        self.dc = dc
        self.parent = parent
        self.row = DATA_IDX
        self.column = 0
        self._label = 'Data'

    def child(self, row):
        if row < len(self.dc):
            return DataItem(self.dc, row, self)

    @property
    def children_count(self):
        return len(self.dc)

    def remove(self):
        return False

    def font(self):
        result = QFont()
        result.setBold(True)
        return result


class DataItem(Item):
    edit_factory = full_edit_factory

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
    def label(self):
        return self.data.label

    @label.setter
    def label(self, value):
        self.data.label = value

    @property
    def style(self):
        return self.data.style

    def remove(self):
        self.dc.remove(self.data)
        return True

    def icon(self):
        return layer_icon(self.data)


class SubsetListItem(Item):

    def __init__(self, dc, parent):
        self.dc = dc
        self.parent = parent
        self.row = SUBSET_IDX
        self._label = 'Subsets'
        self.column = 0

    def child(self, row):
        if row < self.dc.subset_groups:
            return SubsetGroupItem(self.dc, row, self)

    @property
    def children_count(self):
        return len(self.dc.subset_groups)

    def remove(self):
        return False

    def font(self):
        result = QFont()
        result.setBold(True)
        return result


class SubsetGroupItem(Item):
    edit_factory = full_edit_factory

    def __init__(self, dc, row, parent):
        self.parent = parent
        self.dc = dc
        self.row = row
        self.column = 0

    @property
    def subset_group(self):
        return self.dc.subset_groups[self.row]

    @property
    def label(self):
        return self.subset_group.label

    @label.setter
    def label(self, value):
        self.subset_group.label = value

    @property
    def style(self):
        return self.subset_group.style

    @property
    def children_count(self):
        return len(self.subset_group.subsets)

    def remove(self):
        self.dc.remove_subset_group(self.subset_group)
        return True

    def child(self, row):
        return SubsetItem(self.dc, self.subset_group, row, self)

    def icon(self):
        return layer_icon(self.subset_group)


class SubsetItem(Item):
    edit_factory = restricted_edit_factory

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
        sub = self.subset
        return '%s (%s)' % (sub.label, sub.data.label)

    def icon(self):
        return layer_icon(self.subset)

    @property
    def style(self):
        return self.subset.style


class DataCollectionModel(QAbstractItemModel, HubListener):
    def __init__(self, data_collection, parent=None):
        super(DataCollectionModel, self).__init__(parent)
        self.data_collection = data_collection
        self.root = DataCollectionItem(data_collection)
        self._items = {}   # map hashes of Model pointers to model items
                           # without this reference, PySide clobbers instance
                           # data of model items
        self.register_to_hub(self.data_collection.hub)

    def index(self, row, column, parent=QModelIndex()):
        if column != 0:
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root
        else:
            parent_item = self._get_item(parent)
            if parent_item is None:
                return QModelIndex()

        child_item = parent_item.child(row)
        if child_item:
            return self._make_index(row, column, child_item)
        else:
            return QModelIndex()

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

    def flags(self, index=QModelIndex()):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index, role):
        if not index.isValid():
            return

        dispatch = {
            Qt.DisplayRole: self._display_data,
            Qt.FontRole: self._font_data,
            Qt.DecorationRole: self._icon_data,
            Qt.UserRole: self._user_data}

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
        Fetch the QModelIndex for a given data index,
        or the index for the parent data item

        :param data_number: position of data set to fetch, or None
        """
        base = self.index(DATA_IDX, 0)
        if data_number is None:
            return base
        return self.index(data_number, 0, base)

    def subsets_index(self, subset_number=None):
        """
        Fetch the QModelIndex for a given subset,
        or the index for the parent subset item

        :param data_number: position of subset group to fetch, or None
        """
        base = self.index(SUBSET_IDX, 0)
        assert isinstance(self._get_item(base), SubsetListItem)
        if subset_number is None:
            return base
        return self.index(subset_number, 0, base)

    def rowCount(self, index=QModelIndex()):
        if not index.isValid():
            return self.root.children_count

        return self._get_item(index).children_count

    def parent(self, index=None):

        if index is None:  # overloaded QObject.parent()
            return QObject.parent(self)

        item = self._get_item(index)
        if item is None:
            return QModelIndex()

        return self._make_index(item.row, item.column, item.parent)

    def columnCount(self, index):
        return 1

    def removeRow(self, row, parent=QModelIndex()):
        if not parent.isValid():
            return False

        item = self._get_item(parent).child(row)
        if item is None:
            return False

        return item.remove()

    def register_to_hub(self, hub):
        for msg in [m.DataCollectionAddMessage,
                    m.DataCollectionDeleteMessage,
                    m.SubsetCreateMessage,
                    m.SubsetDeleteMessage]:
            hub.subscribe(self, msg, lambda x: self.reset())


class DataCollectionView(QTreeView):

    def __init__(self, data_collection, parent=None):
        super(DataCollectionView, self).__init__(parent)
        self._model = DataCollectionModel(data_collection)
        self.setModel(self._model)

        self.setRootIsDecorated(False)
        self.setExpandsOnDoubleClick(False)
        self.expandToDepth(0)

        self.doubleClicked.connect(self._edit)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.viewport().update)
        self._timer.start(1000)

    def _edit(self, index):
        item = self._model.data(index, role=Qt.UserRole)
        if item.edit_factory is None:
            return

        rect = self.visualRect(index)
        pos = self.mapToGlobal(rect.bottomLeft())
        pos.setY(pos.y() + 1)
        item.edit_factory(pos)


if __name__ == "__main__":
    from glue.qt import get_qapp
    from glue.external.qt.QtGui import QTreeView
    from glue.core import Data, DataCollection
    from glue.qt.widgets.style_dialog import StyleDialog

    app = get_qapp()

    dc = DataCollection([Data(label='x', x=[1, 2, 3]),
                         Data(label='y', y=[1, 2, 3]),
                         Data(label='z', z=[1, 2, 3])])
    sg = dc.new_subset_group()
    sg.label = 'hi'

    view = DataCollectionView(dc)
    view.show()
    view.raise_()
    app.exec_()
