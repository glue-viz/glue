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
    def __init__(self, data=None, parent=None):
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

        self.create_actions()
        self.create_toolbar()
        self.create_menu()
        self.create_secondary_navigator()                
        self._active_layer = None

        cv.ScatterClient.__init__(self, data=data, figure=self.fig)


    def create_actions(self):
        pass

    def create_toolbar(self):
        tb = CloudvizToolbar(self)
        self.toolbar = tb
        self.addToolBar(tb)
        
    def create_menu(self):
        pass

    def create_secondary_navigator(self):
        layout = self.frame.layout()
        bottom = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        xrow, yrow = self.create_variable_droplists()
        tree, row = self.create_layer_tree()
        
        left.addLayout(xrow)
        left.addLayout(yrow)        
        right.addWidget(tree)
        right.addLayout(row)
        right.setContentsMargins(0,0,0,0)

        bottom.addLayout(left)
        bottom.addLayout(right)
        layout.addLayout(bottom)

    def create_variable_droplists(self):
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


        self.xcombo = xcombo
        self.ycombo = ycombo

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

        return xrow, yrow

    def create_layer_tree(self):
        self.tree = {}
        tree = QTreeWidget()
        self.tree['root'] = tree
        tree.setHeaderLabels(["Layers"])

        self.connect(tree, SIGNAL('itemPressed(QTreeWidgetItem *,int)'),
                     self.activate_new_layer)
        self.connect(tree, SIGNAL('itemChanged(QTreeWidgetItem *,int)'), 
                     self.toggle_layer_visibility)

        add = QPushButton(QIcon("icons/plus.png"), "Add")
        subtract = QPushButton(QIcon("icons/minus.png"), "Subtract")
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(subtract)
        row.setContentsMargins(0,0,0,0)
        return tree, row

    def notify_layer_change(self, old, new):
        super(ScatterUI, self).notify_layer_change(old, new)
        if self.toolbar is not None:
            self.toolbar.deselect_roi()
            self.toolbar.set_roi_enabled(isinstance(new, cv.subset.RoiSubset))
        self.tree['root'].setCurrentItem(self.tree[new])

    def notify_data_layer_change(self, old, new):
        super(ScatterUI, self).notify_data_layer_change(old, new)
        self.toolbar.deselect_roi()
        self.set_combos(new)

    def set_combos(self, data):
        if data is None:
            return

        xcombo = self.xcombo
        ycombo = self.ycombo
        fields = self.layers[data]['attributes']
        
        s1 = self.disconnect(xcombo, SIGNAL('currentIndexChanged(int)'), 
                                  self.update_xcombo)
        s2 = self.disconnect(ycombo, SIGNAL('currentIndexChanged(int)'), 
                                   self.update_ycombo)
        xcombo.clear()
        ycombo.clear()
        xcombo.addItems(fields)
        ycombo.addItems(fields)
        xcombo.setCurrentIndex(fields.index(self.layers[data]['x']))
        ycombo.setCurrentIndex(fields.index(self.layers[data]['y']))

        if s1:
            self.connect(xcombo, SIGNAL('currentIndexChanged(int)'), 
                         self.update_xcombo)
        if s2:
            self.connect(ycombo, SIGNAL('currentIndexChanged(int)'), 
                         self.update_ycombo)
            
    def treecheck(self):
        print 'check'
        return

    def update_xcombo(self, **kwargs):
        data = self.active_data
        if data is None: 
            return
        combo = self.xcombo
        att = str(combo.currentText())
        self.set_xdata(att, data=data, **kwargs)

    def update_ycombo(self, **kwargs):
        data = self.active_data
        if data is None:
            return
        combo = self.ycombo
        att = str(combo.currentText())
        self.set_ydata(att, data=data, **kwargs)
    
    def toggle_xlog(self):
        source = self.sender()
        state = source.checkState()
        self.set_xlog(state == Qt.Checked)

    def toggle_ylog(self):
        source = self.sender()
        state = source.checkState()
        self.set_ylog(state == Qt.Checked)

    def toggle_xflip(self):
        source = self.sender()
        state = source.checkState()
        self.set_xflip(state == Qt.Checked)

    def toggle_yflip(self):
        source = self.sender()
        state = source.checkState()
        self.set_yflip(state == Qt.Checked)


    def _add_tree_item(self, item):
        tree = self.tree['root']
        if isinstance(item, cv.Subset):
            d = item.data
            parent = self.tree[d]
            label = item.label
            if label is None:
                ct = parent.childCount()
                datanum = tree.indexOfTopLevelItem(parent)
                label = "Subset %i.%i" % (datanum, ct)
            branch = QTreeWidgetItem(self.tree[d], [label])
        elif isinstance(item, cv.Data):
            label = item.label
            if label is None:
                num = tree.topLevelItemCount()
                label = "Data %i" % num
            branch = QTreeWidgetItem(tree, [label])
        else:
            raise TypeError("Item is not data or subset: %s" % type(item))

        branch.setCheckState(0, Qt.Checked)        
        self.tree[item] = branch
        tree.expandItem(branch)

    def init_layer(self, layer):
        super(ScatterUI, self).init_layer(layer)
        self._add_tree_item(layer)
        
    def activate_new_layer(self):
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
            artist = self.layers[k]['artist']
            artist.set_visible(v.checkState(0) == Qt.Checked)
            self._redraw()

if __name__=="__main__":
    from PyQt4.QtGui import QApplication
    import sys 
    import cloudviz as cv
    from qt_subset_browser_client import QtSubsetBrowserClient

    app = QApplication(sys.argv)

    data = cv.data.TabularData(label="Primary Data Set")
    data2 = cv.data.TabularData(label="Secondary data Set")
    data.read_data('test_table_1.vot', tid=0)
    data2.read_data('test_table_2.vot')
    s = cv.subset.RoiSubset(data, label="First Subset")
    s.style.color = 'red'
    s2 = cv.subset.RoiSubset(data2, label="Second Subset")
    s2.style.color='#aaaaaa'
    
    hub = cv.Hub()
    subset_client = QtSubsetBrowserClient([data, data2])
    scatter_client = ScatterUI([data, data2])
    

    data.register_to_hub(hub)
    data2.register_to_hub(hub)
    subset_client.register_to_hub(hub)
    scatter_client.register_to_hub(hub)

    s.register()
    s2.register()

    subset_client.show()
    scatter_client.show()

    sys.exit(app.exec_())

