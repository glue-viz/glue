from __future__ import absolute_import, division, print_function

from glue.utils.qt.mime import PyMimeData

__all__ = ['GlueItemWidget']


class GlueItemWidget(object):

    """
    A mixin for QtWidgets.QListWidget/GlueTreeWidget subclasses, that provides
    drag+drop funtionality.
    """
    # Implementation detail: QXXWidgetItems are unhashable in PySide,
    # and cannot be used as dictionary keys. we hash on IDs instead

    SUPPORTED_MIME_TYPE = None

    def __init__(self, parent=None):
        super(GlueItemWidget, self).__init__(parent)
        self._mime_data = {}
        self.setDragEnabled(True)

    def mimeTypes(self):
        """
        Return the list of MIME Types supported for this object.
        """
        types = [self.SUPPORTED_MIME_TYPE]
        return types

    def mimeData(self, selected_items):
        """
        Return a list of MIME data associated with the each selected item.

        Parameters
        ----------
        selected_items : list
            A list of ``QtWidgets.QListWidgetItems`` or ``QtWidgets.QTreeWidgetItems`` instances

        Returns
        -------
        result : list
            A list of MIME objects
        """
        try:
            data = [self.get_data(i) for i in selected_items]
        except KeyError:
            data = None
        result = PyMimeData(data, **{self.SUPPORTED_MIME_TYPE: data})

        # apparent bug in pyside garbage collects custom mime
        # data, and crashes. Save result here to avoid
        self._mime = result

        return result

    def get_data(self, item):
        """
        Convenience method to fetch the data associated with a ``QxxWidgetItem``.
        """
        # return item.data(Qt.UserRole)
        return self._mime_data.get(id(item), None)

    def set_data(self, item, data):
        """
        Convenience method to set data associated with a ``QxxWidgetItem``.
        """
        # item.setData(Qt.UserRole, data)
        self._mime_data[id(item)] = data

    def drop_data(self, item):
        self._mime_data.pop(id(item))

    @property
    def data(self):
        return self._mime_data
