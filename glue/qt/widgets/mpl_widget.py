#!/usr/bin/env python

# Python Qt4 bindings for GUI objects
from PyQt4 import QtGui
from PyQt4.QtCore import pyqtSignal

# import the Qt4Agg FigureCanvas object, that binds Figure to
# Qt4Agg backend. It also inherits from QWidget
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# Matplotlib Figure object
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""

    #signals
    rightDrag = pyqtSignal(float, float)
    leftDrag = pyqtSignal(float, float)

    def __init__(self):
        # setup Matplotlib Figure and Axis
        self.fig = Figure(facecolor='#ededed')
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(right=.98, top=.98)

        # initialization of the canvas
        FigureCanvas.__init__(self, self.fig)
        # we define the widget as expandable
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        # notify the system of updated policy
        FigureCanvas.updateGeometry(self)


class MplWidget(QtGui.QWidget):
    """Widget defined in Qt Designer"""

    #signals
    rightDrag = pyqtSignal(float, float)
    leftDrag = pyqtSignal(float, float)

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

if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    win = QtGui.QMainWindow()
    m = MplWidget()
    win.setCentralWidget(m)
    win.show()
    sys.exit(app.exec_())
