#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from functools import wraps

import matplotlib
from matplotlib.figure import Figure

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from qtpy import PYQT5
from glue.utils import DeferredMethod
from glue.config import settings

if PYQT5:
    from matplotlib.backends.backend_qt5 import FigureManagerQT as FigureManager
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
else:
    try:
        from matplotlib.backends.backend_qt4agg import FigureManagerQT as FigureManager
    except ImportError:  # mpl < 1.4
        from matplotlib.backends.backend_qt4agg import FigureManagerQTAgg as FigureManager
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


def defer_draw(func):
    """
    Decorator that globally defers all MplCanvas draw requests until
    function exit.

    If an MplCanvas instance's draw method is invoked multiple times,
    it will only be called once after the wrapped function returns.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            MplCanvas.draw = DeferredMethod(MplCanvas.draw)
            result = func(*args, **kwargs)
        finally:
            MplCanvas.draw.execute_deferred_calls()
            MplCanvas.draw = MplCanvas.draw.original_method
        return result

    return wrapper


class MplCanvas(FigureCanvas):

    """Class to represent the FigureCanvas widget"""

    rightDrag = QtCore.Signal(float, float)
    leftDrag = QtCore.Signal(float, float)
    homeButton = QtCore.Signal()
    resize_begin = QtCore.Signal()
    resize_end = QtCore.Signal()

    def __init__(self):

        self._draw_count = 0
        interactive = matplotlib.is_interactive()
        matplotlib.interactive(False)
        self.roi_callback = None

        self.fig = Figure(facecolor=settings.BACKGROUND_COLOR)

        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)

        FigureCanvas.updateGeometry(self)
        self.manager = FigureManager(self, 0)
        matplotlib.interactive(interactive)

        self._resize_timer = QtCore.QTimer()
        self._resize_timer.setInterval(250)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_timeout)

        self.renderer = None

    def _on_timeout(self):
        buttons = QtWidgets.QApplication.instance().mouseButtons()
        if buttons != Qt.NoButton:
            self._resize_timer.start()
        else:
            self.resize_end.emit()

    def paintEvent(self, event):
        # draw the zoom rectangle more prominently
        try:
            drawRect = self.drawRect
            self.drawRect = False

        except AttributeError:  # mpl  1.4
            drawRect = self._drawRect
            self._drawRect = None

        # super needs this
        if self.renderer is None:
            self.renderer = self.get_renderer()

        super(MplCanvas, self).paintEvent(event)
        if drawRect:
            try:
                x, y, w, h = self.rect[0], self.rect[1], self.rect[2], self.rect[3]
            except TypeError:  # mpl 1.4
                x, y, w, h = drawRect
            p = QtGui.QPainter(self)
            p.setPen(QtGui.QPen(Qt.red, 2, Qt.DotLine))
            p.drawRect(x, y, w, h)
            p.end()

        if self.roi_callback is not None:
            self.roi_callback(self)

    def resizeEvent(self, event):
        if not self._resize_timer.isActive():
            self.resize_begin.emit()
        self._resize_timer.start()
        super(MplCanvas, self).resizeEvent(event)

    def draw(self, *args, **kwargs):
        self._draw_count += 1
        return super(MplCanvas, self).draw(*args, **kwargs)


class MplWidget(QtWidgets.QWidget):

    """Widget defined in Qt Designer"""

    # signals
    rightDrag = QtCore.Signal(float, float)
    leftDrag = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        QtWidgets.QWidget.__init__(self, parent)
        # set the canvas to the Matplotlib widget
        self.canvas = MplCanvas()
        # create a vertical box layout
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.setContentsMargins(0, 0, 0, 0)
        self.vbl.setSpacing(0)
        # add mpl widget to the vertical box
        self.vbl.addWidget(self.canvas)
        # set the layout to the vertical box
        self.setLayout(self.vbl)

        self.canvas.rightDrag.connect(self.rightDrag)
        self.canvas.leftDrag.connect(self.leftDrag)
