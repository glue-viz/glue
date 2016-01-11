from __future__ import absolute_import, division, print_function

from functools import wraps

from glue.external.qt import QtGui
from glue.external.qt.QtCore import Qt

__all__ = ['QComboBoxRecent']

# For the QComboBoxRecent class below, we need to overload methods from the
# original class so that we always apply an offset to indices that are passed as
# input and ones that are returned. To do that, we create wrapper functions.


def subtract_offset(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        index = method(self, *args, **kwargs)
        if index == -1:
            return index
        else:
            return index - self.offset
    return wrapper


def add_offset(method):
    @wraps(method)
    def wrapper(self, index, *args, **kwargs):
        return method(self, index + self.offset, *args, **kwargs)
    return wrapper


def identity(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs)
    return wrapper


class QComboBoxRecent(QtGui.QComboBox):
    """
    Subclass of QComboBox that includes the 5 most recent selections at the
    top.

    In order to make this as compatible as a drop-in replacement, all the usual
    methods are overriden so that the index returned e.g. by currentIndex and
    used to set setCurrentIndex is the one from the original list, not the one
    in the resulting combo box with the most recent items.
    """

    def __init__(self, *args, **kwargs):
        super(QComboBoxRecent, self).__init__(*args, **kwargs)
        self.init_recent()
        self.currentIndexChanged.connect(self._index_changed)
        self.recent = []

    def clear(self):
        super(QComboBoxRecent, self).clear()
        self.init_recent()

    def init_recent(self):
        self._insertItem(0, "Recent items")
        self._setItemData(0, "data", role=Qt.UserRole - 1)
        self._insertItem(1, "All items")
        self._setItemData(1, "data", role=Qt.UserRole - 1)
        self._setCurrentIndex(-1)

    @property
    def offset(self):
        """
        By how many items items are offset in the new combo box compared to the
        original list. This is used to compute offsets for the index.
        """
        return 2 + len(self.recent)

    def _index_changed(self):

        index = self._currentIndex()

        if index == -1:
            return

        self._original_index = index - self.offset

        try:
            self.blockSignals(True)
            text = self._itemText(index)
            data = self._itemData(index)
            for idx in range(len(self.recent)):
                if text == self.recent[idx][0] and data is self.recent[idx][1]:
                    self.recent.pop(idx)
                    self._removeItem(idx + 1)
                    break
            if len(self.recent) == 5:
                self.recent.pop(4)
                self._removeItem(5)
            self.recent.insert(0, (text, data))
            self._insertItem(1, text, data)
            self._setCurrentIndex(1)
        finally:
            self.blockSignals(False)

    # We now overload methods that take the index as one of the arguments, or
    # return an index, and we need to adjust the index accordingly.

    # The following methods return an index and therefore we need to subtract
    # the offset.

    def currentIndex(self):
        index = QtGui.QComboBox.currentIndex(self)
        if index == -1:
            return -1
        elif index < self.offset:
            return self._original_index
        else:
            return index - self.offset

    # currentIndex = subtract_offset(QtGui.QComboBox.currentIndex)
    findData = subtract_offset(QtGui.QComboBox.findData)
    findText = subtract_offset(QtGui.QComboBox.findText)
    count = subtract_offset(QtGui.QComboBox.count)

    # The following methods take an index as the first argument and therefore
    # we need to add an offset.
    insertItem = add_offset(QtGui.QComboBox.insertItem)
    insertItems = add_offset(QtGui.QComboBox.insertItems)
    insertSeparator = add_offset(QtGui.QComboBox.insertSeparator)
    itemData = add_offset(QtGui.QComboBox.itemData)
    itemIcon = add_offset(QtGui.QComboBox.itemIcon)
    itemText = add_offset(QtGui.QComboBox.itemText)
    setItemData = add_offset(QtGui.QComboBox.setItemData)
    removeItem = add_offset(QtGui.QComboBox.removeItem)
    setCurrentIndex = add_offset(QtGui.QComboBox.setCurrentIndex)

    # We now want to keep the original methods accessible with a leading
    # underscore.
    _currentIndex = identity(QtGui.QComboBox.currentIndex)
    _findData = identity(QtGui.QComboBox.findData)
    _findText = identity(QtGui.QComboBox.findText)
    _count = identity(QtGui.QComboBox.count)
    _insertItem = identity(QtGui.QComboBox.insertItem)
    _insertItems = identity(QtGui.QComboBox.insertItems)
    _insertSeparator = identity(QtGui.QComboBox.insertSeparator)
    _itemData = identity(QtGui.QComboBox.itemData)
    _itemIcon = identity(QtGui.QComboBox.itemIcon)
    _itemText = identity(QtGui.QComboBox.itemText)
    _setItemData = identity(QtGui.QComboBox.setItemData)
    _removeItem = identity(QtGui.QComboBox.removeItem)
    _setCurrentIndex = identity(QtGui.QComboBox.setCurrentIndex)
