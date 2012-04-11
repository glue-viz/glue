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
from PyQt4.QtGui import QAction
from PyQt4.QtGui import QActionGroup
from PyQt4 import QtGui

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.colors import ColorConverter
from matplotlib.figure import Figure


import glue
from glue.mpl_scatter_client import MplScatterClient
from glue.scatter_client import ScatterClient
import glue.message as msg
from glue import RasterAxes

class QtScatterClient(QMainWindow, MplScatterClient):

    def __init__(self, data, parent=None, raster=True, **kwargs):

        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Scatter Client")
        self.create_actions()
        self.create_main_frame(data, raster=raster)
        self.create_menu()
        self.create_toolbar()

        MplScatterClient.__init__(self, data, axes=self.axes, **kwargs)

        self.selector = None


    def create_actions(self):
        exit_action = QAction('Close', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(QtGui.qApp.quit)
        self.exit_action = exit_action

        circle = QAction(QIcon("circle.png"), 'Circle', self)
        circle.setStatusTip('Select using a circle')
        circle.setCheckable(True)
        self.circle = circle

        box = QAction(QIcon("square.png"), 'Box', self)
        box.setStatusTip('Select using a box')
        box.setCheckable(True)
        self.box = box

        lasso = QAction(QIcon("lasso.png"), 'lasso', self)
        lasso.setStatusTip('Select using a lasso')
        lasso.setCheckable(True)
        self.lasso = lasso

        move = QAction(QIcon("move.png"), 'move', self)
        move.setStatusTip("Pan axes with left mouse, zoom with right")
        move.setCheckable(True)
        self.move = move

        zoom = QAction(QIcon("zoom.png"), 'zoom', self)
        zoom.setStatusTip("Zoom to Rectangle")
        zoom.setCheckable(True)
        self.zoom = zoom

        self.connect(self.circle, SIGNAL('triggered(bool)'), self.update_selector)
        self.connect(self.box, SIGNAL('triggered(bool)'), self.update_selector)
        self.connect(self.lasso, SIGNAL('triggered(bool)'), self.update_selector)
        self.connect(self.move, SIGNAL('triggered(bool)'), self.mpl_move)
        self.connect(self.zoom, SIGNAL('triggered(bool)'), self.mpl_zoom)

        self.interaction_group = QActionGroup(self)
        self.interaction_group.addAction(circle)
        self.interaction_group.addAction(box)
        self.interaction_group.addAction(lasso)
        self.interaction_group.addAction(move)
        self.interaction_group.addAction(zoom)

        self.interaction_group.setExclusive(True)


    def create_menu(self):
        menu = self.menuBar().addMenu("&File")
        menu.addAction(self.exit_action)

    def create_toolbar(self):
        self.toolbar = self.addToolBar('interaction')
        self.toolbar.addAction(self.circle)
        self.toolbar.addAction(self.box)
        self.toolbar.addAction(self.lasso)
        self.toolbar.addAction(self.move)
        self.toolbar.addAction(self.zoom)


    def on_button_press(self, event):
        self.check_ignore_canvas()

    def on_motion(self, event):
        self.check_ignore_canvas()

    def check_ignore_canvas(self):
        pass

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

    def mpl_move(self):
        pass

    def mpl_zoom(self):
        pass

    def update_selector(self):
        subset = self.data.get_active_subset()
        if not subset:
            return

        if self.selector is not None:
            self.selector.disconnect()
            self.selector = None

        sender = self.sender()

        x = self.get_x_attribute()
        y = self.get_y_attribute()
        if sender is self.box:
            self.selector = glue.roi.MplBoxTool(self.data, x, y,
                                              self.axes)
        elif sender is self.circle:
            self.selector = glue.roi.MplCircleTool(self.data, x, y,
                                                 self.axes)
        elif sender is self.lasso:
            self.selector = glue.roi.MplLassoTool(self.data, x, y,
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

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)

        self.x_component = QComboBox()
        self.y_component = QComboBox()
        for c in data.components:

            # only add numeric columns
            try:
                junk = float(data[c].flat[0])
            except ValueError:
                continue

            self.x_component.addItem(c)
            self.y_component.addItem(c)

        self.connect(self.x_component, SIGNAL('currentIndexChanged(int)'),
                     self.gui_select_x)

        self.connect(self.y_component, SIGNAL('currentIndexChanged(int)'),
                     self.gui_select_y)

        hbox = QHBoxLayout()
        hbox.addWidget(self.x_component)
        hbox.addWidget(self.y_component)
        vbox.addLayout(hbox)

        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
