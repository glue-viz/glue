from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow
from ...core.hub import HubListener

class DataViewer(QMainWindow, HubListener):
    def __init__(self, data, parent=None):
        QMainWindow.__init__(self, parent)
        HubListener.__init__(self)

        self._data = data
        self._hub = None
        self.setAttribute(Qt.WA_DeleteOnClose)

    def register_to_hub(self, hub):
        self._hub = hub

    def unregister(self, hub):
        raise NotImplementedError

    def closeEvent(self, event):
        if self._hub is not None:
            self.unregister(self._hub)
        super(DataViewer, self).closeEvent(event)


