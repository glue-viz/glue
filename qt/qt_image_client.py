import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

import cloudviz as cv
from cloudviz.mpl_image_client import MplImageClient

class QtImageClient(QMainWindow, MplImageClient):

    def __init__(self, data, parent=None, **kwargs):
        
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Image Client")
        
        self.create_main_frame(data)
        MplImageClient.__init__(self, data, axes=self.axes, **kwargs)
        
        self.selector = None # ROI tool to use
        

    def on_button_press(self, event):
        self.check_ignore_canvas()
        
        
    def on_motion(self, event):
        self.check_ignore_canvas()
        
    def check_ignore_canvas(self):
        if self.mpl_toolbar.mode != '':
            if self.selector is not None:
                self.selector.disconnect()
                self.selector = None
                
    def on_push(self):
        subset = self.data.get_active_subset()
        if not subset:
            return

        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None

        sender = self.sender()
        if sender is self.boxWidget:
            if not isinstance(subset, cv.subset.ElementSubset):
                return
            self.selector = cv.roi.MplBoxTool(subset, 'XPIX', 'YPIX', 
                                              self.axes)
        elif sender is self.circleWidget:
            if not isinstance(subset, cv.subset.ElementSubset):
                return
            self.selector = cv.roi.MplCircleTool(subset, 'XPIX', 'YPIX',
                                                 self.axes)
        elif sender is self.lassoWidget:
            if not isinstance(subset, cv.subset.ElementSubset):
                return
            self.selector = cv.roi.MplLassoTool(subset, 'XPIX', 'YPIX',
                                                self.axes)

        elif sender is self.treeWidget:
            if not isinstance(subset, cv.subset.TreeSubset):
                return
            self.selector = cv.roi.MplTreeTool(subset, self.axes)
            
    def select_component(self):
        component = str(self.componentWidget.currentItem().text())
        self.set_component(component)

    def create_main_frame(self, data):
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

        hbox = QHBoxLayout()
        
        self.boxWidget = QPushButton("Box")
        self.circleWidget = QPushButton("Circle")
        self.lassoWidget = QPushButton("Lasso")
        self.treeWidget = QPushButton("Tree-Based Selection")
        self.componentWidget = QListWidget()
        for c in data.components:
            self.componentWidget.addItem(c)
        self.connect(self.componentWidget, SIGNAL('itemSelectionChanged()'), 
                     self.select_component)
        

        #self.connect(self.boxWidget, SIGNAL('clicked()'), self.on_push)
        #self.connect(self.circleWidget, SIGNAL('clicked()'), self.on_push)
        #self.connect(self.lassoWidget, SIGNAL('clicked()'), self.on_push)
        self.connect(self.treeWidget, SIGNAL('clicked()'), self.on_push)

        #hbox.addWidget(self.boxWidget)
        #hbox.addWidget(self.circleWidget)
        #hbox.addWidget(self.lassoWidget)
        hbox.addWidget(self.treeWidget)
        vbox.addLayout(hbox)
        vbox.addWidget(self.componentWidget)
        
        
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)

        

        
