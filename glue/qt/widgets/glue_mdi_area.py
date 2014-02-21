from ...external.qt import QtGui
from ...external.qt.QtCore import Qt

from ... import core
from ..mime import LAYER_MIME_TYPE, LAYERS_MIME_TYPE


class GlueMdiArea(QtGui.QMdiArea):
    """Glue's MdiArea implementation.

    Drop events with :class:`~glue.core.Data` objects in
    :class:`~glue.qt.mime.PyMimeData` load these objects into new
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
        self.setBackground(QtGui.QBrush(QtGui.QColor(250, 250, 250)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._show_help = False

    @property
    def show_help(self):
        return self._show_help

    @show_help.setter
    def show_help(self, value):
        self._show_help = value
        self.repaint()

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
            if isinstance(layer, core.data.Data):
                self._application.choose_new_data_viewer(layer)
            else:
                assert isinstance(layer, core.subset.Subset)
                self._application.choose_new_data_viewer(layer.data)

        if md.hasFormat(LAYER_MIME_TYPE):
            new_layer(md.data(LAYER_MIME_TYPE))

        assert md.hasFormat(LAYERS_MIME_TYPE)
        for layer in md.data(LAYERS_MIME_TYPE):
            new_layer(layer)

        event.accept()

    def mousePressEvent(self, event):
        """Right mouse press in the MDI area opens a new data viewer"""
        if event.button() != Qt.RightButton:
            return
        self._application.choose_new_data_viewer()

    def close(self):
        self.closeAllSubWindows()
        super(GlueMdiArea, self).close()

    def paintEvent(self, event):
        super(GlueMdiArea, self).paintEvent(event)
        if (not self.show_help):
            return

        painter = QtGui.QPainter(self.viewport())
        painter.setPen(QtGui.QColor(210, 210, 210))
        font = painter.font()
        font.setPointSize(48)
        font.setWeight(font.Black)
        painter.setFont(font)
        rect = self.contentsRect()
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter,
                         "Drag Data To Plot")
