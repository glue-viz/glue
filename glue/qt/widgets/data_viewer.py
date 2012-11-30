import os

from PyQt4.QtGui import (QMainWindow, QMessageBox, QWidget)

from PyQt4.QtCore import Qt

from ...core.hub import HubListener
from ...core.data import Data
from ...core.subset import Subset
from ..layer_artist_model import QtLayerArtistContainer, LayerArtistView
from .. import get_qapp
from ..mime import LAYERS_MIME_TYPE, LAYER_MIME_TYPE


class DataViewer(QMainWindow, HubListener):
    """Base class for all Qt DataViewer widgets.

    This defines a minimal interface, and implemlements the following:

       * An automatic call to unregister on window close
       * Drag and drop support for adding data
    """
    def __init__(self, data, parent=None):
        QMainWindow.__init__(self, parent)
        HubListener.__init__(self)
        self.setWindowIcon(get_qapp().windowIcon())
        self._data = data
        self._hub = None
        self._container = QtLayerArtistContainer()
        self._view = LayerArtistView()
        self._view.setModel(self._container.model)
        self._tb_vis = {}  # store whether toolbars are enabled
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setAnimated(False)
        self._toolbars = []
        self._warn_close = True
        self.setContentsMargins(2, 2, 2, 2)

    def register_to_hub(self, hub):
        self._hub = hub

    def unregister(self, hub):
        """ Abstract method to unsubscribe from messages """
        raise NotImplementedError

    def add_data(self, data):
        """ Add a data instance to the viewer

        This must be overridden by a subclass

        :param data: Data object to add
        :type data: :class:`~glue.core.Data`
        """
        raise NotImplementedError

    def add_subset(self, subset):
        """ Add a subset to the viewer

        This must be overridden by a subclass

        :param subset: Subset instance to add
        :type subset: :class:`~glue.core.subset.Subset`
        """
        raise NotImplementedError

    def dragEnterEvent(self, event):
        """ Accept the event if it has data layers"""
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.accept()
        elif event.mimeData().hasFormat(LAYERS_MIME_TYPE):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Add layers to the viewer if contained in mime data """
        def add_layer(layer):
            if isinstance(layer, Data):
                self.add_data(layer)
            else:
                assert isinstance(layer, Subset)
                self.add_subset(layer)

        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            add_layer(event.mimeData().data(LAYER_MIME_TYPE))

        assert event.mimeData().hasFormat(LAYERS_MIME_TYPE)

        for layer in event.mimeData().data(LAYERS_MIME_TYPE):
            add_layer(layer)

        event.accept()

    def mousePressEvent(self, event):
        """ Consume mouse press events, and prevent them from propagating
            down to the MDI area """
        event.accept()

    def close(self, warn=True):
        self._warn_close = warn
        super(DataViewer, self).close()
        self._warn_close = True

    def closeEvent(self, event):
        """ Call unregister on window close """
        if not self._confirm_close():
            event.ignore()
            return

        if self._hub is not None:
            self.unregister(self._hub)
        super(DataViewer, self).closeEvent(event)

    def _confirm_close(self):
        """Ask for close confirmation

        :rtype: bool. True if user wishes to close. False otherwise
        """
        if self._warn_close and (not os.environ.get('GLUE_TESTING')):
            buttons = QMessageBox.Ok | QMessageBox.Cancel
            dialog = QMessageBox.warning(self, "Confirm Close",
                                         "Do you want to close this window?",
                                         buttons=buttons,
                                         defaultButton=QMessageBox.Cancel)
            return dialog == QMessageBox.Ok
        return True

    def _confirm_large_data(self, data):
        warn_msg = ("WARNING: Data set has %i points, and may render slowly."
                    " Continue?" % data.size)
        title = "Add large data set?"
        ok = QMessageBox.Ok
        cancel = QMessageBox.Cancel
        buttons = ok | cancel
        result = QMessageBox.question(self, title, warn_msg,
                                      buttons=buttons,
                                      defaultButton=cancel)
        return result == ok

    def layer_view(self):
        return self._view

    def options_widget(self):
        return QWidget()

    def addToolBar(self, tb):
        super(DataViewer, self).addToolBar(tb)
        self._toolbars.append(tb)

    def show_toolbars(self):
        """Re-enable any toolbars that were hidden with `hide_toolbars()`

        Does not re-enable toolbars that were hidden by other means
        """
        for tb in self._toolbars:
            if self._tb_vis.get(tb, False):
                tb.setVisible(True)

    def hide_toolbars(self):
        """ Hide all the toolbars in the viewer.

        This action can be reversed by calling `show_toolbars()`
        """
        for tb in self._toolbars:
            self._tb_vis[tb] = self._tb_vis.get(tb, False) or tb.isVisible()
            tb.setVisible(False)

    def set_focus(self, state):
        if state:
            css = """
            DataViewer
            {
            border: 2px solid;
            border-color: rgb(56, 117, 215);
            }
            """
            self.setStyleSheet(css)
            self.show_toolbars()
        else:
            css = """
            DataViewer
            {
            border: none;
            }
            """
            self.setStyleSheet(css)
            self.hide_toolbars()
