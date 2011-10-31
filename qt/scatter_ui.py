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
from cloudviz_toolbar import CloudvizToolbar

class ScatterUI(QMainWindow, cv.ScatterClient):
    def __init__(self, data, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Scatter Plot")

        self.frame = QWidget()
        self.setCentralWidget(self.frame)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.frame.setLayout(layout)
        
        dpi = 60
        self.fig = Figure((7.0, 6.0), dpi = dpi, facecolor='#ededed')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)
        self.frame.layout().addWidget(self.canvas)
        cv.ScatterClient.__init__(self, data, figure=self.fig)

        self.create_actions()
        self.create_toolbar()
        self.create_menu()
        self.create_secondary_navigator()        
        
        self.active_layer = None

    @property
    def active_layer(self):
        return self._active_layer

    @active_layer.setter
    def active_layer(self, layer):
        if layer is not None and layer not in self.layers:
            raise TypeError("Invalid layer")
        self._active_layer = layer
        isSubset = isinstance(layer, cv.subset.RoiSubset)
        self.toolbar.buttons['circle'].setDisabled(not isSubset)
        self.toolbar.buttons['box'].setDisabled(not isSubset)
        self.toolbar.buttons['lasso'].setDisabled(not isSubset)
        if not isSubset:
            self.toolbar.buttons['circle'].setChecked(False)
            self.toolbar.buttons['box'].setChecked(False)
            self.toolbar.buttons['lasso'].setChecked(False)
            self.toolbar._active = None
    def create_actions(self):
        pass

    def create_toolbar(self):
        tb = CloudvizToolbar(self)
        self.toolbar = tb
        print self.toolbar
        self.addToolBar(tb)
        

    def create_plot_window(self):
        self.dpi = 60
        self.fig = Figure((7.0, 6.0), dpi=self.dpi, facecolor='#ededed')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)

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
            c = [c for c in data.components if
                 np.can_cast(data[c].dtype, np.float)]
            self.combo_data[d] = {'fields': c, 'x': 0, 'y': 1}
        self.set_combos(self.data)
        left.addLayout(xrow)
        left.addLayout(yrow)

        self.connect(xcombo, SIGNAL('currentIndexChanged(int)'), 
                     self.update_xcombo)
        self.connect(ycombo, SIGNAL('currentIndexChanged(int)'), 
                     self.update_ycombo)
        self.connect(xlog, SIGNAL('stateChanged(int)'), 
                     self.toggle_xlog)
        self.connect(ylog, SIGNAL('stateChanged(int)'),
                     self.toggle_ylog)
        self.connect(xflip, SIGNAL('stateChanged(int)'),
                     self.toggle_xflip)
        self.connect(yflip, SIGNAL('stateChanged(int)'),
                     self.toggle_yflip)

        # layer tree display
        right = QVBoxLayout()
        self.tree = {}
        tree = QTreeWidget()
        tree.setHeaderLabels(["Layers"])
        items = []
        for i,d in enumerate(self._data):
            label = d.label
            if label is None:
                label = "Data %i" % i
            item = QTreeWidgetItem([label])
            self.tree[d] = item
            items.append(item)
            for j, s in enumerate(d.subsets):
                label = s.label
                if label is None:
                    label = "Subset %i.%i" % (i, j)
                item = QTreeWidgetItem(items[-1], [label])
                self.tree[s] = item
        tree.addTopLevelItems(items)
        tree.expandAll()
        self.connect(tree, SIGNAL('itemPressed(QTreeWidgetItem *,int)'),
                     self.highlight_layer)
        self.connect(tree, SIGNAL('itemChanged(QTreeWidgetItem *,int)'), 
                     self.toggle_layer_visibility)

        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            iterator.value().setCheckState(0, Qt.Checked)
            iterator += 1

        add = QPushButton(QIcon("icons/plus.png"), "Add")
        subtract = QPushButton(QIcon("icons/minus.png"), "Subtract")
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
            self.combo_data[current]['x'] = x
            self.combo_data[current]['y'] = y

        self.combo_data['current'] = data
        fields = self.combo_data[data]['fields']
        xcombo.clear()
        ycombo.clear()
        xcombo.addItems(fields)
        ycombo.addItems(fields)
        xcombo.setCurrentIndex(self.combo_data[data]['x'])
        ycombo.setCurrentIndex(self.combo_data[data]['y'])
        self.update_xcombo()
        self.update_ycombo()

    def treecheck(self):
        print 'check'
        return

    def update_xcombo(self):
        data = self.combo_data['current']
        combo = self.combo_data['xcombo']
        att = str(combo.currentText())
        self.set_xdata(att, data=data)

    def update_ycombo(self):
        print 'update_ycombo'
        data = self.combo_data['current']
        combo = self.combo_data['ycombo']
        att = str(combo.currentText())
        print att
        self.set_ydata(att, data=data)
    
    def toggle_xlog(self):
        source = self.sender()
        state = source.checkState()
        mode = 'linear' if state == Qt.Unchecked else 'log'
        self.ax.set_xscale(mode)
        self._redraw()

    def toggle_ylog(self):
        source = self.sender()
        state = source.checkState()
        mode = 'linear' if state == Qt.Unchecked else 'log'
        self.ax.set_yscale(mode)
        self._redraw()

    def toggle_xflip(self):
        source = self.sender()
        state = source.checkState()
        flip = state == Qt.Checked
        range = self.ax.get_xlim()
        if flip:
            self.ax.set_xlim(max(range), min(range))
        else:
            self.ax.set_xlim(min(range), max(range))        
        self._redraw()

    def toggle_yflip(self):
        source = self.sender()
        state = source.checkState()
        flip = state == Qt.Checked
        range = self.ax.get_ylim()
        if flip:
            self.ax.set_ylim(max(range), min(range))
        else:
            self.ax.set_ylim(min(range), max(range))        
        self._redraw()

    def _add_subset(self, message):
        super(ScatterUI, self)._add_subset(message)
        s = message.sender
        d = s.data
        parent = self.tree[d]
        ct = parent.childCount()
        datanum = int(str(parent.text(0)).split()[-1])
        label = "Subset %i.%i" % (datanum, ct)
        item = QTreeWidgetItem(self.tree[d], [label])
        self.tree[s] = item
        item.setCheckState(0, Qt.Checked)

    def highlight_layer(self):
        """ When the user clicks on a layer in the layer tree,
        set the active_layer to match the selection"""
        sender = self.sender()
        item = sender.currentItem()
        for k,v in self.tree.iteritems():
            if v is not item: continue
            self.active_layer = k
        
    def toggle_layer_visibility(self):
        sender = self.sender()
        item = sender.currentItem()
        for k,v in self.tree.iteritems():
            if v is not item: continue
            self.layers[k]['artist'].set_visible(v.checkState(0) == Qt.Checked)
            self._redraw()

if __name__=="__main__":
    from PyQt4.QtGui import QApplication
    import sys 
    import cloudviz as cv
    from qt_subset_browser_client import QtSubsetBrowserClient

    app = QApplication(sys.argv)

    data = cv.data.TabularData()
    data.read_data('../examples/oph_c2d_yso_catalog.tbl')
    s = cv.subset.RoiSubset(data)
    s.style.color = 'red'
    
    hub = cv.Hub()
    subset_client = QtSubsetBrowserClient(data)
    scatter_client = ScatterUI(data)
    

    data.register_to_hub(hub)
    subset_client.register_to_hub(hub)
    scatter_client.register_to_hub(hub)

    s.register()
    
    subset_client.show()
    scatter_client.show()

    sys.exit(app.exec_())

