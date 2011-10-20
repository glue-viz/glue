from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import Qt
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ColorConverter
from matplotlib.figure import Figure
import numpy as np

import cloudviz as cv
from cloudviz.mpl_scatter_client import MplScatterClient
from cloudviz.scatter_client import ScatterClient
import cloudviz.message as msg
from cloudviz import RasterAxes

class ScatterUI(QMainWindow, cv.Client):
    def __init__(self, data, parent=None):
        QMainWindow.__init__(self, parent)
        cv.Client.__init__(self, data)

        self.frame = QWidget()
        self.setCentralWidget(self.frame)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.frame.setLayout(layout)
        
        self.setWindowTitle("Scatter Plot")

        self.create_plot_window()
        self.create_actions()
        self.create_toolbar()
        self.create_menu()
        self.create_secondary_navigator()

    def create_actions(self):
        pass

    def create_toolbar(self):
        self.toolbar = CloudvizToolbar(self.canvas, self.frame)

    def create_plot_window(self):
        self.dpi = 60
        self.fig = Figure((7.0, 6.0), dpi=self.dpi, facecolor='#ededed')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)

        ax = self.fig.add_subplot(111)
        ax.scatter((np.nan), (np.nan))
        self._ax = ax
        
        self.frame.layout().addWidget(self.canvas)

        
    def create_menu(self):
        pass

    def create_secondary_navigator(self):
        layout = self.frame.layout()
        bottom = QHBoxLayout()
        left = QVBoxLayout()

        # select which variables to plot
        xrow = QHBoxLayout()
        xlabel = QLabel("x axis")
        xcombo = QComboBox()
        xlog = QCheckBox("log")
        xflip = QCheckBox("flip")
        xrow.addWidget(xlabel)
        xrow.addWidget(xcombo)
        xrow.addWidget(xlog)
        xrow.addWidget(xflip)

        yrow = QHBoxLayout()
        ylabel = QLabel("y axis")
        ycombo = QComboBox()
        ylog = QCheckBox("log")
        yflip = QCheckBox("flip")
        yrow.addWidget(ylabel)
        yrow.addWidget(ycombo)
        yrow.addWidget(ylog)
        yrow.addWidget(yflip)
        
        self.combo_data = {}
        self.combo_data['current'] = None
        self.combo_data['xcombo'] = xcombo
        self.combo_data['ycombo'] = ycombo
        for d in self._data:
            c = [c in data.components where 
                 np.can_cast(data[c].dtype, np.float)]
            self.combo_data[d] = {'fields': c, 'x':, 0, 'y':, 1}
        self.set_combos(self.data)
        left.addLayout(xrow)
        left.addLayout(yrow)
            

        # layer display
        right = QVBoxLayout()        
        tree = QTreeWidget()
        tree.setHeaderLabels(["Layers"])
        items = []
        for i,d in enumerate(self._data):
            label = d.label
            if label is None:
                label = "Data %i" % i
            items.append(QTreeWidgetItem([label]))
            for j, s in enumerate(d.subsets):
                label = s.label
                if label is None:
                    label = "Subset %i.%i" % (i, j)
                QTreeWidgetItem(items[-1], [label])
        tree.addTopLevelItems(items)

        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            iterator.value().setCheckState(0, Qt.Checked)
            iterator += 1

        add = QPushButton(QIcon("icons/plus.svg"), "Add")
        subtract = QPushButton(QIcon("icons/minus.svg"), "Subtract")
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(subtract)
        row.setContentsMargins(0,0,0,0)
        
        right.addWidget(tree)
        right.addLayout(row)
        right.setContentsMargins(0,0,0,0)
        bottom.addLayout(left)
        bottom.addLayout(right)
        layout.addLayout(bottom)

    def set_combos(self, data):
        current = self.combo_data['current']
        xcombo = self.combo_data['xcombo']
        ycombo = self.combo_data['ycombo']

        if current is not None:
            x = xcombo.currentIndex()
            y = ycombo.currentIndex()
            self.combo_data[current]]['x'] = x
            self.combo_data[current]['y'] = y

        self.combo_data['current'] = data
        fields = self.combo_data[data]['fields']
        xcombo.clear()
        ycombo.clear()
        xcombo.addItems(fields)
        ycombo.addItems(fields)
        xcombo.setCurrentIndex(self.combo_data[data]['x'])
        ycombo.setCurrentIndex(self.combo_data[data]['y'])

    def treecheck(self):
        print 'check'
        return

if __name__=="__main__":
    from PyQt4.QtGui import QApplication
    import sys 
    app = QApplication(sys.argv)
    
    gui = ScatterUI()
    gui.show()
    sys.exit(app.exec_())

