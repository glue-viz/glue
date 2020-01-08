import os

from qtpy.QtCore import Qt
from qtpy import QtCore, QtWidgets
from glue.utils.qt import get_qapp
from glue.core.qt.mime import LAYERS_MIME_TYPE, LAYER_MIME_TYPE

__all__ = ['BaseQtViewerWidget']


class BaseQtViewerWidget(QtWidgets.QMainWindow):
    """
    Base Qt class for all DataViewer widgets. This is not a viewer class in
    itself but is the base widget that should be used for any Qt viewer that
    is to appear inside the MDI area.
    """

    window_closed = QtCore.Signal()
    toolbar_added = QtCore.Signal()

    _closed = False

    def __init__(self, parent=None):
        """
        :type session: :class:`~glue.core.session.Session`
        """

        super(BaseQtViewerWidget, self).__init__(parent)

        self.setWindowIcon(get_qapp().windowIcon())

        status_bar = self.statusBar()
        status_bar.setSizeGripEnabled(False)
        status_bar.setStyleSheet("QStatusBar{font-size:10px}")

        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setAnimated(False)
        self.setContentsMargins(2, 2, 2, 2)

        self._mdi_wrapper = None  # GlueMdiSubWindow that self is embedded in
        self._warn_close = True

    def dragEnterEvent(self, event):
        """
        Accept drag-and-drop of data or subset objects.
        """
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.accept()
        elif event.mimeData().hasFormat(LAYERS_MIME_TYPE):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Accept drag-and-drop of data or subset objects.
        """
        if event.mimeData().hasFormat(LAYER_MIME_TYPE):
            self.request_add_layer(event.mimeData().data(LAYER_MIME_TYPE))

        assert event.mimeData().hasFormat(LAYERS_MIME_TYPE)

        for layer in event.mimeData().data(LAYERS_MIME_TYPE):
            self.request_add_layer(layer)

        event.accept()

    def mousePressEvent(self, event):
        """
        Consume mouse press events, and prevent them from propagating
        down to the MDI area.
        """
        event.accept()

    def close(self, warn=True):

        if self._closed:
            return

        if warn and not self._confirm_close():
            return

        self._warn_close = False

        if getattr(self, '_mdi_wrapper', None) is not None:
            self._mdi_wrapper.close()
            self._mdi_wrapper = None
        else:
            try:
                QtWidgets.QMainWindow.close(self)
            except RuntimeError:
                # In some cases the above can raise a "wrapped C/C++ object of
                # type ... has been deleted" error, in which case we can just
                # ignore and carry on.
                pass

        self._closed = True

    def mdi_wrap(self):
        """
        Wrap this object in a GlueMdiSubWindow
        """
        from glue.app.qt.mdi_area import GlueMdiSubWindow
        sub = GlueMdiSubWindow()
        sub.setWidget(self)
        self.destroyed.connect(sub.close)
        sub.resize(self.size())
        self._mdi_wrapper = sub
        return sub

    @property
    def position(self):
        """
        The location of the viewer as a tuple of ``(x, y)``
        """
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
            QtWidgets.QMainWindow.move(self, x, y)

    @property
    def viewer_size(self):
        """
        Size of the viewer as a tuple of ``(width, height)``
        """
        if self._mdi_wrapper is not None:
            sz = self._mdi_wrapper.size()
        else:
            sz = self.size()
        return sz.width(), sz.height()

    @viewer_size.setter
    def viewer_size(self, value):
        width, height = value
        if self._mdi_wrapper is None:
            self.resize(width, height)
        else:
            self._mdi_wrapper.resize(width, height)

    def closeEvent(self, event):
        """
        Call unregister on window close
        """

        if self._warn_close and not self._confirm_close():
            event.ignore()
            return

        super(BaseQtViewerWidget, self).closeEvent(event)
        event.accept()

        self.window_closed.emit()

    def isVisible(self):
        # Override this so as to catch RuntimeError: wrapped C/C++ object of
        # type ... has been deleted
        try:
            return self.isVisible()
        except RuntimeError:
            return False

    def _confirm_close(self):
        """Ask for close confirmation

        :rtype: bool. True if user wishes to close. False otherwise
        """
        if self._warn_close and not os.environ.get('GLUE_TESTING'):
            buttons = QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            dialog = QtWidgets.QMessageBox.warning(self, "Confirm Close",
                                                   "Do you want to close this window?",
                                                   buttons=buttons,
                                                   defaultButton=QtWidgets.QMessageBox.Cancel)
            return dialog == QtWidgets.QMessageBox.Ok
        return True

    def layer_view(self):
        return QtWidgets.QWidget()

    def options_widget(self):
        return QtWidgets.QWidget()

    def set_focus(self, state):
        if state:
            css = """
            DataViewer
            {
            border: 2px solid;
            border-color: rgb(56, 117, 215);
            }
            """
        else:
            css = """
            DataViewer
            {
            border: none;
            }
            """
        self.setStyleSheet(css)

    @property
    def window_title(self):
        return str(self)

    def update_window_title(self):
        try:
            self.setWindowTitle(self.window_title)
        except RuntimeError:  # Avoid C/C++ errors when closing viewer
            pass

    def set_status(self, message):
        sb = self.statusBar()
        sb.showMessage(message)
