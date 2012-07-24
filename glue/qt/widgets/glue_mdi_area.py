from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from ... import core


class GlueMdiArea(QtGui.QMdiArea):
    """Glue's MdiArea implementation.

    Drop events with :class:`~glue.core.Data` objects in
    :class:`~glue.qt.qtutil.PyMimeData` load these objects into new
    data viewers
    """
    def __init__(self, application, parent=None):
        """
        :param application: The Glue application to which this is attached
        :type application: :class:`~glue.qt.glue_application.GlueApplication`
        """
        super(GlueMdiArea, self).__init__(parent)
        self._application = application
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def dragEnterEvent(self, event):
        """ Accept the event if it has an application/py_instance format """
        if event.mimeData().hasFormat('application/py_instance'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Load a new data viewer if the event has a glue Data object """
        obj = event.mimeData().data('application/py_instance')
        if not isinstance(obj, core.data.Data):
            return
        self._application.new_data_viewer(obj)

    def mousePressEvent(self, event):
        """Right mouse press in the MDI area opens a new data viewer"""
        if event.button() != Qt.RightButton:
            return
        self._application.new_data_viewer()

    def close(self):
        self.closeAllSubWindows()
        super(GlueMdiArea, self).close()
