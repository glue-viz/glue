from PyQt4 import QtGui
from PyQt4.QtCore import Qt

from ... import core
from .. import glue_qt_resources  # pylint: disable=W0611


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
        if event.mimeData().hasFormat('application/py_instance'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Load a new data viewer if the event has a glue Data object """
        obj = event.mimeData().data('application/py_instance')
        if isinstance(obj, core.data.Data):
            self._application.new_data_viewer(obj)
        elif isinstance(obj, core.subset.Subset):
            self._application.new_data_viewer(obj.data)

    def mousePressEvent(self, event):
        """Right mouse press in the MDI area opens a new data viewer"""
        if event.button() != Qt.RightButton:
            return
        self._application.new_data_viewer()

    def close(self):
        self.closeAllSubWindows()
        super(GlueMdiArea, self).close()

    def paintEvent(self, event):
        super(GlueMdiArea, self).paintEvent(event)
        if len(self.subWindowList()) != 0 or (not self.show_help):
            return

        painter = QtGui.QPainter(self.viewport())
        painter.setPen(QtGui.QColor(210, 210, 210))
        font = painter.font()
        font.setPointSize(48)
        font.setWeight(font.Black)
        painter.setFont(font)
        rect = event.rect()
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter,
                         "Drag Data To Plot")
