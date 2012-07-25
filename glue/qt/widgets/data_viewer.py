from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow

from ...core.hub import HubListener
from ...core.data import Data


class DataViewer(QMainWindow, HubListener):
    """Base class for all Qt DataViewer widgets.

    This defines a minimal interface, and implemlements the following:

       * An automatic call to unregister on window close
       * Drag and drop support for adding data
    """
    def __init__(self, data, parent=None):
        QMainWindow.__init__(self, parent)
        HubListener.__init__(self)

        self._data = data
        self._hub = None
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)

    def register_to_hub(self, hub):
        self._hub = hub

    def unregister(self, hub):
        """ Abstract method to unsubscribe from messages """
        raise NotImplementedError

    def add_data(self, data):
        """ Abstract method to add data objects to the viewer
        :param data: Data object to add
        :type data: :class:`~glue.core.Data`
        """
        raise NotImplementedError

    def dragEnterEvent(self, event):
        """ Accept the event if it has an application/py_instance format """
        if event.mimeData().hasFormat('application/py_instance'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Add data to the viewer if the event has a glue Data object """
        obj = event.mimeData().data('application/py_instance')
        if not isinstance(obj, Data):
            return
        self.add_data(obj)

    def mousePressEvent(self, event):
        """ Consume mouse press events, and prevent them from propagating
            down to the MDI area """
        event.accept()

    def closeEvent(self, event):
        """ Call unregister on window close """
        if self._hub is not None:
            self.unregister(self._hub)
        super(DataViewer, self).closeEvent(event)
