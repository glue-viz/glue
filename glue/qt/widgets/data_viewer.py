import os

from ...external.qt.QtGui import (
    QMainWindow, QMessageBox, QWidget, QMdiSubWindow)

from ...external.qt.QtCore import Qt

from ...core.application_base import ViewerBase
from ..decorators import set_cursor

from ..layer_artist_model import QtLayerArtistContainer, LayerArtistView
from .. import get_qapp
from ..mime import LAYERS_MIME_TYPE, LAYER_MIME_TYPE

__all__ = ['DataViewer']


class DataViewer(QMainWindow, ViewerBase):

    """Base class for all Qt DataViewer widgets.

    This defines a minimal interface, and implemlements the following::

       * An automatic call to unregister on window close
       * Drag and drop support for adding data
    """
    _container_cls = QtLayerArtistContainer

    def __init__(self, session, parent=None):
        """
        :type session: :class:`~glue.core.Session`
        """
        QMainWindow.__init__(self, parent)
        ViewerBase.__init__(self, session)
        self.setWindowIcon(get_qapp().windowIcon())
        self._view = LayerArtistView()
        self._view.setModel(self._container.model)
        self._tb_vis = {}  # store whether toolbars are enabled
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setAnimated(False)
        self._toolbars = []
        self._warn_close = True
        self.setContentsMargins(2, 2, 2, 2)
        self._mdi_wrapper = None  # QMdiSubWindow that self is embedded in

    def remove_layer(self, layer):
        self._container.pop(layer)

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

        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            self.request_add_layer(event.mimeData().data(LAYER_MIME_TYPE))

        assert event.mimeData().hasFormat(LAYERS_MIME_TYPE)

        for layer in event.mimeData().data(LAYERS_MIME_TYPE):
            self.request_add_layer(layer)

        event.accept()

    def mousePressEvent(self, event):
        """ Consume mouse press events, and prevent them from propagating
            down to the MDI area """
        event.accept()

    apply_roi = set_cursor(Qt.WaitCursor)(ViewerBase.apply_roi)

    def close(self, warn=True):
        self._warn_close = warn
        super(DataViewer, self).close()
        self._warn_close = True

    def mdi_wrap(self):
        """Wrap this object in a QMdiSubWindow"""
        sub = QMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub

        return sub

    @property
    def position(self):
        target = self._mdi_wrapper or self
        pos = target.pos()
        return pos.x(), pos.y()

    def move(self, x=None, y=None):
        x0, y0 = self.position
        if x is None:
            x = x0
        if y is None:
            y = y0
        if self._mdi_wrapper is not None:
            self._mdi_wrapper.move(x, y)
        else:
            QMainWindow.move(self, x, y)

    @property
    def viewer_size(self):
        sz = QMainWindow.size(self)
        return sz.width(), sz.height()

    @viewer_size.setter
    def viewer_size(self, value):
        width, height = value
        self.resize(width, height)
        if self._mdi_wrapper is not None:
            self._mdi_wrapper.resize(width, height)

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
        self._tb_vis[tb] = True

    def show_toolbars(self):
        """Re-enable any toolbars that were hidden with `hide_toolbars()`

        Does not re-enable toolbars that were hidden by other means
        """
        for tb in self._toolbars:
            if self._tb_vis.get(tb, False):
                tb.setEnabled(True)

    def hide_toolbars(self):
        """ Disable all the toolbars in the viewer.

        This action can be reversed by calling `show_toolbars()`
        """
        for tb in self._toolbars:
            self._tb_vis[tb] = self._tb_vis.get(tb, False) or tb.isVisible()
            tb.setEnabled(False)

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
