from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow
from PyQt4.QtGui import QPushButton
from PyQt4.QtGui import QHBoxLayout
from PyQt4.QtGui import QCheckBox
from PyQt4.QtGui import QComboBox
from PyQt4.QtGui import QVBoxLayout
from PyQt4.QtGui import QWidget

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

import glue
from glue.mpl_image_client import MplImageClient

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
                #uncheck check boxes
                self.treeWidget.setChecked(Qt.Unchecked)


    def on_push(self):
        subset = self.data.get_active_subset()
        if not subset:
            return

        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None

        sender = self.sender()

        if sender.checkState() == Qt.Unchecked:
            return

        if sender is self.boxWidget:
            self.selector = glue.roi.MplBoxTool(self.data, 'XPIX', 'YPIX',
                                              self.axes)
        elif sender is self.circleWidget:
            self.selector = glue.roi.MplCircleTool(self.data, 'XPIX', 'YPIX',
                                                 self.axes)
        elif sender is self.lassoWidget:
            self.selector = glue.roi.MplLassoTool(self.data, 'XPIX', 'YPIX',
                                                self.axes)

        elif sender is self.treeWidget:
            self.selector = glue.roi.MplTreeTool(self.data, 'XPIX', 'YPIX',
                                               self.axes)

    def select_component(self):
        component = str(self.componentWidget.currentText())
        MplImageClient.set_component(self, component)

    def set_component(self, component):
        # update combo box, which will take care of rest
        index = self.componentWidget.findText(component)
        if index == -1:
            raise KeyError("Not a valid component: %s" %component)

        self.componentWidget.setCurrentIndex(index)

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
        self.treeWidget = QCheckBox("Tree-Based Selection")
        self.componentWidget = QComboBox()
        for c in data.components:
            self.componentWidget.addItem(c)
        self.connect(self.componentWidget, SIGNAL('currentIndexChanged(int)'),
                     self.select_component)


        #self.connect(self.boxWidget, SIGNAL('clicked()'), self.on_push)
        #self.connect(self.circleWidget, SIGNAL('clicked()'), self.on_push)
        #self.connect(self.lassoWidget, SIGNAL('clicked()'), self.on_push)
        self.connect(self.treeWidget, SIGNAL('stateChanged(int)'), self.on_push)

        #hbox.addWidget(self.boxWidget)
        #hbox.addWidget(self.circleWidget)
        #hbox.addWidget(self.lassoWidget)
        hbox.addWidget(self.treeWidget)
        hbox.addWidget(self.componentWidget)
        vbox.addLayout(hbox)


        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)




