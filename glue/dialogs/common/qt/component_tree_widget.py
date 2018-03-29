from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt

__all__ = ['ComponentTreeWidget']


class ComponentTreeWidget(QtWidgets.QTreeWidget):

    order_changed = QtCore.Signal()

    def select_cid(self, cid):
        for item in self:
            if item.data(0, Qt.UserRole) is cid:
                self.select_item(item)
                return
        raise ValueError("Could not find find cid {0} in list".format(cid))

    def select_item(self, item):
        self.selection = self.selectionModel()
        self.selection.select(QtCore.QItemSelection(self.indexFromItem(item, 0),
                                                    self.indexFromItem(item, self.columnCount() - 1)),
                              QtCore.QItemSelectionModel.ClearAndSelect)

    @property
    def selected_item(self):
        items = self.selectedItems()
        return items[0] if len(items) == 1 else None

    @property
    def selected_cid(self):
        selected = self.selected_item
        return None if selected is None else selected.data(0, Qt.UserRole)

    def add_cid_and_label(self, cid, columns, editable=True):
        item = QtWidgets.QTreeWidgetItem(self.invisibleRootItem(), columns)
        item.setData(0, Qt.UserRole, cid)
        if editable:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setFlags(item.flags() ^ Qt.ItemIsDropEnabled)

    def __iter__(self):
        root = self.invisibleRootItem()
        for idx in range(root.childCount()):
            yield root.child(idx)

    def __len__(self):
        return self.invisibleRootItem().childCount()

    def dropEvent(self, event):
        selected = self.selected_item
        super(ComponentTreeWidget, self).dropEvent(event)
        self.select_item(selected)
        self.order_changed.emit()

    def mousePressEvent(self, event):
        self.clearSelection()
        super(ComponentTreeWidget, self).mousePressEvent(event)
