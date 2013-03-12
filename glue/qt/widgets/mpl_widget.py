#!/usr/bin/env python

# Python Qt4 bindings for GUI objects
from ...external.qt import QtGui
from ...external.qt.QtCore import Signal, Qt

# import the Qt4Agg FigureCanvas object, that binds Figure to
# Qt4Agg backend. It also inherits from QWidget
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as \
    FigureCanvas
from matplotlib.backends.backend_qt4agg import FigureManagerQTAgg as \
    FigureManager

# Matplotlib Figure object
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""

    #signals
    rightDrag = Signal(float, float)
    leftDrag = Signal(float, float)

    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure(facecolor='#ffffff')

        try:
            self.fig.set_tight_layout(True)
        except AttributeError:  # matplotlib < 1.1
            pass

        # initialization of the canvas
        FigureCanvas.__init__(self, self.fig)

        # we define the widget as expandable
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)

        # notify the system of updated policy
        FigureCanvas.updateGeometry(self)
        self.manager = FigureManager(self, 0)

    def paintEvent(self, event):
        #draw the zoom rectangle more prominently
        drawRect = self.drawRect
        self.drawRect = False
        super(MplCanvas, self).paintEvent(event)
        if drawRect:
            p = QtGui.QPainter(self)
            p.setPen(QtGui.QPen(Qt.red, 2, Qt.DotLine))
            p.drawRect(self.rect[0], self.rect[1], self.rect[2], self.rect[3])
            p.end()


class MplWidget(QtGui.QWidget):
    """Widget defined in Qt Designer"""

    #signals
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
