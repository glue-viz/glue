#!/usr/bin/env python
from functools import partial, wraps

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


class DeferredMethod(object):

    """
    This class stubs out a method, and provides a
    callable interface that logs its calls. These
    can later be actually executed on the original (non-stubbed)
    method by calling executed_deferred_calls
    """

    def __init__(self, method):
        self.method = method
        self.calls = []  # avoid hashability issues with dict/set

    @property
    def original_method(self):
        return self.method

    def __call__(self, instance, *a, **k):
        if instance not in (c[0] for c in self.calls):
            self.calls.append((instance, a, k))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self.__call__, instance)

    def execute_deferred_calls(self):
        for instance, args, kwargs in self.calls:
            self.method(instance, *args, **kwargs)


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

        self.renderer = None

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
