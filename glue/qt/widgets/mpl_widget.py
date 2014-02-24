#!/usr/bin/env python

from ...external.qt import QtGui
from ...external.qt.QtCore import Signal, Qt, QTimer

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas

try:
    from matplotlib.backends.backend_qt4agg import FigureManagerQTAgg as \
        FigureManager
except ImportError:  # mpl >= 1.4
    from matplotlib.backends.backend_qt4agg import FigureManagerQT as \
        FigureManager

import matplotlib

from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    """Class to represent the FigureCanvas widget"""

    rightDrag = Signal(float, float)
    leftDrag = Signal(float, float)
    homeButton = Signal()
    resize_begin = Signal()
    resize_end = Signal()

    def __init__(self):
        interactive = matplotlib.is_interactive()
        matplotlib.interactive(False)
        self.roi_callback = None

        self.fig = Figure(facecolor='#ffffff')
        try:
            self.fig.set_tight_layout(True)
        except AttributeError:  # matplotlib < 1.1
            pass

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)
        self.manager = FigureManager(self, 0)
        matplotlib.interactive(interactive)

        self._resize_timer = QTimer()
        self._resize_timer.setInterval(250)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_timeout)

        self._draw_timer = QTimer()
        self._draw_timer.setInterval(10)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.timeout.connect(super(MplCanvas, self).draw)

        self.renderer = None

    def draw(self):
        # delay a bit before actually drawing,
        # to avoid multiple rapid draws
        self._draw_timer.start()

    def _on_timeout(self):
        buttons = QtGui.QApplication.instance().mouseButtons()
        if buttons != Qt.NoButton:
            self._resize_timer.start()
        else:
            self.resize_end.emit()

    def paintEvent(self, event):
        # draw the zoom rectangle more prominently
        drawRect = self.drawRect
        self.drawRect = False

        # super needs this
        if self.renderer is None:
            self.renderer = self.get_renderer()

        super(MplCanvas, self).paintEvent(event)
        if drawRect:
            p = QtGui.QPainter(self)
            p.setPen(QtGui.QPen(Qt.red, 2, Qt.DotLine))
            p.drawRect(self.rect[0], self.rect[1], self.rect[2], self.rect[3])
            p.end()

        if self.roi_callback is not None:
            self.roi_callback(self)

    def resizeEvent(self, event):
        if not self._resize_timer.isActive():
            self.resize_begin.emit()
        self._resize_timer.start()
        super(MplCanvas, self).resizeEvent(event)


class MplWidget(QtGui.QWidget):

    """Widget defined in Qt Designer"""

    # signals
    rightDrag = Signal(float, float)
    leftDrag = Signal(float, float)

    def __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        QtGui.QWidget.__init__(self, parent)
        # set the canvas to the Matplotlib widget
        self.canvas = MplCanvas()
        # create a vertical box layout
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.setContentsMargins(0, 0, 0, 0)
        self.vbl.setSpacing(0)
        # add mpl widget to the vertical box
        self.vbl.addWidget(self.canvas)
        # set the layout to the vertical box
        self.setLayout(self.vbl)

        self.canvas.rightDrag.connect(self.rightDrag)
        self.canvas.leftDrag.connect(self.leftDrag)
