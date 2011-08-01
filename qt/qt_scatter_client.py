from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow 
from PyQt4.QtGui import QComboBox 
from PyQt4.QtGui import QPushButton
from PyQt4.QtGui import QHBoxLayout
from PyQt4.QtGui import QCheckBox
from PyQt4.QtGui import QVBoxLayout
from PyQt4.QtGui import QColorDialog
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QButtonGroup
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QColor
from PyQt4.QtGui import QIcon

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.colors import ColorConverter
from matplotlib.figure import Figure


import cloudviz as cv
from cloudviz.mpl_scatter_client import MplScatterClient
from cloudviz.scatter_client import ScatterClient
import cloudviz.message as msg
from cloudviz import RasterAxes

class QtScatterClient(QMainWindow, MplScatterClient):
    
    def __init__(self, data, parent=None, raster=True, **kwargs):

        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Scatter Client")

        self.create_main_frame(data, raster=raster)
        MplScatterClient.__init__(self, data, axes=self.axes, **kwargs)

        self.selector = None

    def on_button_press(self, event):
        self.check_ignore_canvas()

    def on_motion(self, event):
        self.check_ignore_canvas()

    def check_ignore_canvas(self):
        if self.mpl_toolbar.mode != '':
            if self.selector is not None:
                self.selector.disconnect()
                self.selector = None
    
    def _set_attribute(self, attribute, axis=None):
        ScatterClient._set_attribute(self, attribute, axis)
        
        if axis == 'x':
            if self.selector is not None:
                self.selector.set_x_attribute(attribute)
            index = self.x_component.findText(attribute)
            self.x_component.setCurrentIndex(index)
        elif axis == 'y':
            if self.selector is not None:
                self.selector.set_y_attribute(attribute)
            index = self.y_component.findText(attribute)
            self.y_component.setCurrentIndex(index)
        else:
            raise AttributeError("axis must be x or y")

    def gui_select_x(self):
        sender = self.sender()
        component = str(self.x_component.currentText())
        self.set_xdata(component)

    def gui_select_y(self):
        sender = self.sender()
        component = str(self.y_component.currentText())
        self.set_ydata(component)

    def select_selector(self):
        subset = self.data.get_active_subset()
        if not subset:
            return

        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None

        sender = self.sender()

        if sender.checkState() == Qt.Unchecked:
            return

        x = self.get_x_attribute()
        y = self.get_y_attribute()
        if sender is self.boxWidget:
            self.selector = cv.roi.MplBoxTool(self.data, x, y, 
                                              self.axes)
        elif sender is self.circleWidget:
            self.selector = cv.roi.MplCircleTool(self.data, x, y,
                                                 self.axes)
        elif sender is self.lassoWidget:
            self.selector = cv.roi.MplLassoTool(self.data, x, y,
                                                self.axes)

    def create_main_frame(self, data, raster=True):
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi = self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        if raster:
            self.axes = RasterAxes(self.fig, [0.1, 0.1, 0.8, 0.8])
            self.fig.add_axes(self.axes)
        else:
            self.axes = self.fig.add_subplot(111)

        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        
        self.boxWidget = QCheckBox("Box")
        self.circleWidget = QCheckBox("Circle")
        self.lassoWidget = QCheckBox("Lasso")
        
        self.x_component = QComboBox()
        self.y_component = QComboBox()
        for c in data.components:
            self.x_component.addItem(c)
            self.y_component.addItem(c)

        self.connect(self.x_component, SIGNAL('currentIndexChanged(int)'), 
                     self.gui_select_x)

        self.connect(self.y_component, SIGNAL('currentIndexChanged(int)'), 
                     self.gui_select_y)
        
        self.connect(self.boxWidget, SIGNAL('stateChanged(int)'), self.select_selector)
        self.connect(self.circleWidget, SIGNAL('stateChanged(int)'), self.select_selector)
        self.connect(self.lassoWidget, SIGNAL('stateChanged(int)'), self.select_selector)

        group = QButtonGroup()
        group.setExclusive(True)
        group.addButton(self.boxWidget)
        group.addButton(self.circleWidget)
        group.addButton(self.lassoWidget)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.boxWidget)
        hbox.addWidget(self.circleWidget)
        hbox.addWidget(self.lassoWidget)
        vbox.addLayout(hbox)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.x_component)
        hbox.addWidget(self.y_component)
        vbox.addLayout(hbox)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
