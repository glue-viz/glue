from __future__ import absolute_import, division, print_function

import os

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui
from glue.core.application_base import ViewerBase
from glue.core.qt.layer_artist_model import QtLayerArtistContainer, LayerArtistWidget
from glue.external.qt import get_qapp
from glue.core.qt.mime import LAYERS_MIME_TYPE, LAYER_MIME_TYPE
from glue.utils.qt import set_cursor


__all__ = ['DataViewer']


class DataViewer(ViewerBase, QtGui.QMainWindow):

    """Base class for all Qt DataViewer widgets.

    This defines a minimal interface, and implemlements the following::

       * An automatic call to unregister on window close
       * Drag and drop support for adding data
    """
    _layer_artist_container_cls = QtLayerArtistContainer
    _layer_style_widget_cls = None

    LABEL = 'Override this'

    def __init__(self, session, parent=None):
        """
        :type session: :class:`~glue.core.Session`
        """
        QtGui.QMainWindow.__init__(self, parent)
        ViewerBase.__init__(self, session)
        self.setWindowIcon(get_qapp().windowIcon())
        self._view = LayerArtistWidget(layer_style_widget_cls=self._layer_style_widget_cls)
        self._view.layer_list.setModel(self._layer_artist_container.model)
        self._tb_vis = {}  # store whether toolbars are enabled
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setAnimated(False)
        self._toolbars = []
        self._warn_close = True
        self.setContentsMargins(2, 2, 2, 2)
        self._mdi_wrapper = None  # GlueMdiSubWindow that self is embedded in
        self.statusBar().setStyleSheet("QStatusBar{font-size:10px}")

        # close window when last plot layer deleted
        self._layer_artist_container.on_empty(lambda: self.close(warn=False))
        self._layer_artist_container.on_changed(self.update_window_title)

    def remove_layer(self, layer):
        self._layer_artist_container.pop(layer)

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
        """Wrap this object in a GlueMdiSubWindow"""
        from glue.app.qt.mdi_area import GlueMdiSubWindow
        sub = GlueMdiSubWindow()
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

    @position.setter
    def position(self, xy):
        x, y = xy
        self.move(x, y)

    def move(self, x=None, y=None):
        """
        Move the viewer to a new XY pixel location

        You can also set the position attribute to a new tuple directly.

        Parameters
        ----------
        x : int (optional)
           New x position
        y : int (optional)
           New y position
        """
        x0, y0 = self.position
        if x is None:
            x = x0
        if y is None:
            y = y0
        if self._mdi_wrapper is not None:
            self._mdi_wrapper.move(x, y)
        else:
            QtGui.QMainWindow.move(self, x, y)

    @property
    def viewer_size(self):
        if self._mdi_wrapper is not None:
            sz = self._mdi_wrapper.size()
        else:
            sz = self.size()
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

        self._layer_artist_container.clear_callbacks()
        self._layer_artist_container.clear()

        super(DataViewer, self).closeEvent(event)
        event.accept()

    def _confirm_close(self):
        """Ask for close confirmation

        :rtype: bool. True if user wishes to close. False otherwise
        """
        if self._warn_close and (not os.environ.get('GLUE_TESTING')) and self.isVisible():
            buttons = QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel
            dialog = QtGui.QMessageBox.warning(self, "Confirm Close",
                                               "Do you want to close this window?",
                                               buttons=buttons,
                                               defaultButton=QtGui.QMessageBox.Cancel)
            return dialog == QtGui.QMessageBox.Ok
        return True

    def _confirm_large_data(self, data):
        warn_msg = ("WARNING: Data set has %i points, and may render slowly."
                    " Continue?" % data.size)
        title = "Add large data set?"
        ok = QtGui.QMessageBox.Ok
        cancel = QtGui.QMessageBox.Cancel
        buttons = ok | cancel
        result = QtGui.QMessageBox.question(self, title, warn_msg,
                                            buttons=buttons,
                                            defaultButton=cancel)
        return result == ok

    def layer_view(self):
        return self._view

    def options_widget(self):
        return QtGui.QWidget()

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

    def __str__(self):
        return self.LABEL

    def unregister(self, hub):
        """
        Override to perform cleanup operations when disconnecting from hub
        """
        pass

    @property
    def window_title(self):
        return str(self)

    def update_window_title(self):
        self.setWindowTitle(self.window_title)
