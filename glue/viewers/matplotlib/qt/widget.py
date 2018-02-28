#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import warnings
import matplotlib
from matplotlib.figure import Figure

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from glue.config import settings

from matplotlib.backends.backend_qt5 import FigureManagerQT as FigureManager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# We want to ignore warnings about left==right and bottom==top since these are
# not critical and the default behavior makes sense.
warnings.filterwarnings('ignore', '.*Attempting to set identical left==right', UserWarning)
warnings.filterwarnings('ignore', '.*Attempting to set identical bottom==top', UserWarning)


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

        self._draw_zoom_rect = None

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

        # super needs this
        if self.renderer is None:
            self.renderer = self.get_renderer()

        super(MplCanvas, self).paintEvent(event)

        # See drawRectangle for what _draw_zoom_rect is
        if self._draw_zoom_rect is not None:
            painter = QtGui.QPainter(self)
            self._draw_zoom_rect(painter)
            painter.end()

        if self.roi_callback is not None:
            self.roi_callback(self)

    def drawRectangle(self, rect):

        # The default zoom rectangle in Matplotlib is quite faint, and there is
        # no easy mechanism for changing the default appearance. However, the
        # drawing of the zoom rectangle is done in the public method
        # drawRectangle on the canvas. This method sets up a callback function
        # that is then called during paintEvent. However, we shouldn't use the
        # same private attribute since this might break, so we use a private
        # attribute with a different name, which means the one matplotlib uses
        # will remain empty and not plot anything.

        if rect is None:
            _draw_zoom_rect = None
        else:
            def _draw_zoom_rect(painter):
                pen = QtGui.QPen(QtGui.QPen(Qt.red, 2, Qt.DotLine))
                painter.setPen(pen)
                try:
                    dpi_ratio = self.devicePixelRatio() or 1
                except AttributeError:  # Matplotlib <2
                    dpi_ratio = 1
                painter.drawRect(*(pt / dpi_ratio for pt in rect))

        # This function will be called at the end of the paintEvent
        self._draw_zoom_rect = _draw_zoom_rect

        # We need to call update to force the canvas to be painted again
        self.update()

    def resizeEvent(self, event):
        if not self._resize_timer.isActive():
            self.resize_begin.emit()
        self._resize_timer.start()
        super(MplCanvas, self).resizeEvent(event)

    def draw(self, *args, **kwargs):
        self._draw_count += 1
        return super(MplCanvas, self).draw(*args, **kwargs)

    def keyPressEvent(self, event):
        event.setAccepted(False)
        super(MplCanvas, self).keyPressEvent(event)


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
