from __future__ import absolute_import, division, print_function

import weakref

from qtpy.QtCore import Qt
from qtpy import QtCore, QtGui, QtWidgets
from glue import core
from glue.core.qt.mime import LAYER_MIME_TYPE, LAYERS_MIME_TYPE


class GlueMdiArea(QtWidgets.QMdiArea):

    """Glue's MdiArea implementation.

    Drop events with :class:`~glue.core.data.Data` objects in
    :class:`~glue.utils.qt.PyMimeData` load these objects into new
    data viewers
    """

    def __init__(self, application, parent=None):
        """
        :param application: The Glue application to which this is attached
        :type application: :class:`~glue.app.qt.application.GlueApplication`
        """
        super(GlueMdiArea, self).__init__(parent)
        self._application = weakref.ref(application)
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setBackground(QtGui.QBrush(QtGui.QColor(250, 250, 250)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def addSubWindow(self, sub):
        super(GlueMdiArea, self).addSubWindow(sub)
        self.repaint()

    def dragEnterEvent(self, event):
        """ Accept the event if it has an application/py_instance format """

        if event.mimeData().hasFormat(LAYERS_MIME_TYPE):
            event.accept()
        elif event.mimeData().hasFormat(LAYER_MIME_TYPE):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Load a new data viewer if the event has a glue Data object """
        md = event.mimeData()

        def new_layer(layer):
            application = self._application()
            if application is None:
                return
            if isinstance(layer, core.data.Data):
                application.choose_new_data_viewer(layer)
            else:
                assert isinstance(layer, core.subset.Subset)
                application.choose_new_data_viewer(layer.data)

        if md.hasFormat(LAYER_MIME_TYPE):
            new_layer(md.data(LAYER_MIME_TYPE))

        assert md.hasFormat(LAYERS_MIME_TYPE)
        for layer in md.data(LAYERS_MIME_TYPE):
            new_layer(layer)

        event.accept()

    def close(self):
        self.closeAllSubWindows()
        super(GlueMdiArea, self).close()

    def paintEvent(self, event):
        super(GlueMdiArea, self).paintEvent(event)

        painter = QtGui.QPainter(self.viewport())
        painter.setPen(QtGui.QColor(210, 210, 210))
        font = painter.font()
        font.setPointSize(font.pointSize() * 4)
        font.setWeight(font.Black)
        painter.setFont(font)
        rect = self.contentsRect()
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter,
                         "Drag Data To Plot")


class GlueMdiSubWindow(QtWidgets.QMdiSubWindow):
    closed = QtCore.Signal()

    def closeEvent(self, event):
        super(GlueMdiSubWindow, self).closeEvent(event)
        self.closed.emit()
