import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

import cloudviz as cv
from cloudviz.mpl_tree_client import MplTreeClient
from cloudviz.tree_layout import DendrogramLayout
from cloudviz.io import extract_data_fits
import cloudviz.data_dendro_cpp as cpp

class QtTreeClient(QMainWindow, MplTreeClient):

    def __init__(self, data, parent=None, **kwargs):

        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Tree Client')
        
        self.create_main_frame()
        MplTreeClient.__init__(self, data, axes=self.axes, **kwargs)
        self._do_select = True

    def on_button_press(self, event):
        if self.mpl_toolbar.mode != '':
            return

        self._do_select = not self._do_select
        
    def on_motion(self, event):
        if self.mpl_toolbar.mode != '':
            return

        if not self._do_select: return

        if not event.inaxes:
            return

        subset = self.data.get_active_subset()
        if not subset:
            return
        
        x = event.xdata
        y = event.ydata
        branch = self.layout.pick(x,y)
        if not branch:
            subset.node_list = []
        else:
            id = branch.get_subtree_indices()
            subset.node_list = id

    def create_main_frame(self):
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi = self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axes = self.fig.add_subplot(111)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)


        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

