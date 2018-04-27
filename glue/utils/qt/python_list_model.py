from __future__ import absolute_import, division, print_function

from qtpy import QtCore
from qtpy.QtCore import Qt

__all__ = ['PythonListModel']


class PythonListModel(QtCore.QAbstractListModel):

    """
    A Qt Model that wraps a python list, and exposes a list-like interface

    This can be connected directly to multiple QListViews, which will
    stay in sync with the state of the container.
    """

    def __init__(self, items, parent=None):
        """
        Create a new model

        Parameters
        ----------
        items : list
            The initial list to wrap
        parent : QObject
            The model parent
        """
        super(PythonListModel, self).__init__(parent)
        self.items = items

    def rowCount(self, parent=None):
        """Number of rows"""
        return len(self.items)

    def headerData(self, section, orientation, role):
        """Column labels"""
        if role != Qt.DisplayRole:
            return None
        return "%i" % section

    def row_label(self, row):
        """ The textual label for the row"""
        return str(self.items[row])

    def data(self, index, role):
        """Retrieve data at each index"""
        if not index.isValid():
            return None
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.row_label(index.row())
        if role == Qt.UserRole:
            return self.items[index.row()]

    def setData(self, index, value, role):
        """
        Update the data in-place

        Parameters
        ----------
        index : QModelIndex
            The location of the change
        value : object
            The new value
        role : QEditRole
            Which aspect of the model to update
        """
        if not index.isValid():
            return False

        if role == Qt.UserRole:
            row = index.row()
            self.items[row] = value
            self.dataChanged.emit(index, index)
            return True

        return super(PythonListModel, self).setDdata(index, value, role)

    def removeRow(self, row, parent=None):
        """
        Remove a row from the table

        Parameters
        ----------
        row : int
            Row to remove

        Returns
        -------
        successful : bool
        """
        if row < 0 or row >= len(self.items):
            return False

        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        self._remove_row(row)
        self.endRemoveRows()
        return True

    def pop(self, row=None):
        """
        Remove and return an item (default last item)

        Parameters
        ----------
        row : int (optional)
            Which row to remove. Default=last

        Returns
        --------
        popped : object
        """
        if row is None:
            row = len(self) - 1
        result = self[row]
        self.removeRow(row)
        return result

    def _remove_row(self, row):
        # actually remove data. Subclasses can override this as needed
        self.items.pop(row)

    def __getitem__(self, row):
        return self.items[row]

    def __setitem__(self, row, value):
        index = self.index(row)
        self.setData(index, value, role=Qt.UserRole)

    def __len__(self):
        return len(self.items)

    def insert(self, row, value):
        self.beginInsertRows(QtCore.QModelIndex(), row, row)
        self.items.insert(row, value)
        self.endInsertRows()
        self.rowsInserted.emit(self.index(row), row, row)

    def append(self, value):
        row = len(self)
        self.insert(row, value)

    def extend(self, values):
        for v in values:
            self.append(v)

    def set_list(self, values):
        """
        Set the model to a new list
        """
        self.beginResetModel()
        self.items = values
        self.endResetModel()
