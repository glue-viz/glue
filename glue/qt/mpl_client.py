#!/usr/bin/env python

import sys, os, random
from PyQt4 import QtGui, QtCore

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

from ..clients.simple_mpl_client import Ui_MainWindow


class ScatterCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, table, parent=None, width=5, height=4, dpi=100):

        # Initialize the figure
        fig = Figure(figsize=(width, height), dpi=dpi)

        # Initialize the axes
        self.ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # We want the axes cleared every time plot() is called
        self.ax.hold(False)

        # Save table
        self.table = table

        # Set default columns to show
        self.column_names = []
        for column in self.table.columns:
            if self.table.columns[column].dtype in [float, np.float32, np.float64]:
                self.column_names.append(column)
        self.col_x = self.column_names[0]
        self.col_y = self.column_names[1]

        # Initialize canvas
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        # Set Qt parameters
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.update_scatter()

    def update_x(self, index):
        self.col_x = self.column_names[index]
        self.update_scatter()

    def update_y(self, index):
        self.col_y = self.column_names[index]
        self.update_scatter()

    def update_scatter(self):

        if hasattr(self, '_scatter'):
            self._scatter.set_offsets(zip(self.table[self.col_x], self.table[self.col_y]))
        else:
            self._scatter = self.ax.scatter(self.table[self.col_x], self.table[self.col_y])

        self.ax.set_xlim(self.table[self.col_x].min(), self.table[self.col_x].max())
        self.ax.set_ylim(self.table[self.col_y].min(), self.table[self.col_y].max())

        self.draw()


class ScatterApplication(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self, table):

        # Save table
        self.table = table

        # Initialize main window and UI
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)

        # Set Qt parameters
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Table Scatter Client")

        # Set behavior of File -> Quit
        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # Add scatter client to MPL widget holder
        l = QtGui.QVBoxLayout(self.mpl_view)
        self.scatter = ScatterCanvas(table, parent=self.mpl_view, width=5, height=4, dpi=100)
        l.addWidget(self.scatter)

        # Add columns to x and y axis
        variant = QtCore.QVariant()
        for column in self.scatter.column_names:
            self.combo_xaxis.addItem(QtCore.QString(column), variant)
            self.combo_yaxis.addItem(QtCore.QString(column), variant)

        self.connect(self.combo_xaxis, QtCore.SIGNAL('activated(int)'), self.scatter.update_x)
        self.connect(self.combo_yaxis, QtCore.SIGNAL('activated(int)'), self.scatter.update_y)

    def file_quit(self):
        self.close()

    def close_event(self, ce):
        self.file_quit()

qApp = QtGui.QApplication(sys.argv)

import atpy
t = atpy.Table('aj285677t3_votable.xml')

aw = ScatterApplication(t)
aw.show()

sys.exit(qApp.exec_())
