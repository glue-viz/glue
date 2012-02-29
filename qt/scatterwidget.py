from PyQt4.QtCore import Qt, SIGNAL
from PyQt4.QtGui import *

from ui_scatterwidget import Ui_ScatterWidget
import numpy as np

import cloudviz as cv
import cloudviz.message as msg
from cloudviz.scatter_client import ScatterClient

from qtutil import mpl_to_qt4_color, qt4_to_mpl_color

class ScatterWidget(QWidget, cv.HubListener) :
    def __init__(self, data):
        QWidget.__init__(self)
        self.ui = Ui_ScatterWidget()
        self.ui.setupUi(self)
        self.client = ScatterClient(data,
                                    self.ui.mplWidget.canvas.fig,
                                    self.ui.mplWidget.canvas.ax)
        self.layer_dict = {}
        self.populate_layer_tree()
        self.connect()
        self.set_combos(self.client.data[0])

    def connect(self):
        ui = self.ui
        cl = self.client
        ui.xLogCheckBox.stateChanged.connect(lambda x: cl.set_xlog(x==Qt.Checked))
        ui.yLogCheckBox.stateChanged.connect(lambda x: cl.set_ylog(x==Qt.Checked))
        ui.xFlipCheckBox.stateChanged.connect(lambda x: cl.set_xflip(x==Qt.Checked))
        ui.yFlipCheckBox.stateChanged.connect(lambda x: cl.set_yflip(x==Qt.Checked))
        ui.xAxisComboBox.currentIndexChanged.connect(self.update_xatt)
        ui.yAxisComboBox.currentIndexChanged.connect(self.update_yatt)

        ui.layerAddButton.pressed.connect(self.dummy)
        ui.layerRemoveButton.pressed.connect(self.dummy)
        ui.linkButton.pressed.connect(self.dummy)

        ui.layerTree.itemSelectionChanged.connect(self.set_active_layer_to_layer_tree)
        ui.layerTree.itemChanged.connect(self.toggle_layer_visibility)
        ui.layerTree.itemClicked.connect(self.edit_layer)

    def set_active_layer_to_layer_tree(self):
        current = self.ui.layerTree.currentItem()
        self.set_active_layer(current)

    def register_to_hub(self, hub):
        self.client.register_to_hub(hub)
        data_filt = lambda x:x.sender.data in self.client.data
        dc_filt = lambda x: x.sender is self.client._data
        hub.subscribe(self,
                      msg.SubsetCreateMessage,
                      handler=lambda x:self.init_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.SubsetUpdateMessage,
                      handler=lambda x:self.sync_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.SubsetDeleteMessage,
                      handler=lambda x:self.remove_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.DataCollectionActiveChange,
                      handler=lambda x:self.set_active_layer(self.layer_dict[x.sender.active]),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataCollectionActiveDataChange,
                      handler=lambda x:self.set_combos(x.sender.active.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataCollectionAddMessage,
                      handler=lambda x:self.init_layer(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=lambda x:self.remove_layer(x.sender),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=lambda x:self.sync_layer(x.sender),
                      filter=data_filt)

    def set_active_layer(self, layer):
        self.client.data.active = self.layer_dict[layer]
        self.ui.layerTree.setCurrentItem(layer)

    def update_xatt(self, index, **kwargs):
        data = self.client.data.active_data
        if data is None:
            return
        combo = self.ui.xAxisComboBox
        att = str(combo.currentText())
        self.client.set_xdata(att, data=data, **kwargs)

    def update_yatt(self, index, **kwargs):
        data = self.client.data.active_data
        if data is None:
            return
        combo = self.ui.yAxisComboBox
        att = str(combo.currentText())
        self.client.set_ydata(att, data=data, **kwargs)

    def edit_layer(self):
        """ Handle events from user modifying point properties in the layer tree """
        column = self.ui.layerTree.currentColumn()
        item = self.ui.layerTree.currentItem()
        if item is None: return
        layer = self.layer_dict[item]

        if column == 1:
            #update color
            dialog = QColorDialog()
            initial = mpl_to_qt4_color(layer.style.color)
            color = dialog.getColor(initial = initial)
            layer.style.color = qt4_to_mpl_color(color)
        elif column == 2:
            # update symbol
            dialog = QInputDialog()
            symb, ok = dialog.getItem(None, 'Pick a Symbol', 'Pick a Symbol', ['.', 'o', 'v', '>', '<', '^'])
            if ok: layer.style.marker = symb
        elif column == 3:
            #update point size
            dialog = QInputDialog()
            size, ok = dialog.getInt(None, 'Point Size', 'Point Size', value = layer.style.markersize,
                                      min = 1, max = 1000, step = 1)
            if ok: layer.style.markersize = size

    def set_combos(self, data):
        """ Update the attribute combo boxes to reflect the currently-edited data set """

        if data is None:
            return
        print 'setting combos'
        xcombo = self.ui.xAxisComboBox
        ycombo = self.ui.yAxisComboBox
        fields = self.client.get_attributes(data)

        xcombo.currentIndexChanged.disconnect(self.update_xatt)
        ycombo.currentIndexChanged.disconnect(self.update_yatt)

        xcombo.clear()
        ycombo.clear()
        xcombo.addItems(fields)
        ycombo.addItems(fields)
        xcombo.setCurrentIndex(fields.index(self.client.current_x_attribute(data)))
        ycombo.setCurrentIndex(fields.index(self.client.current_y_attribute(data)))

        xcombo.currentIndexChanged.connect(self.update_xatt)
        ycombo.currentIndexChanged.connect(self.update_yatt)

    def add_tree_item(self, item):
        """ Add a new object to the layer tree, and display it.

        Parameters
        ----------
        item : data or subset object
        """
        tree = self.ui.layerTree
        if isinstance(item, cv.Subset):
            d = item.data
            parent = self.layer_dict[d]
            label = item.label
            if label is None:
                ct = parent.childCount()
                datanum = tree.indexOfTopLevelItem(parent)
                label = "Subset %i.%i" % (datanum, ct)
            parent = self.layer_dict[d]
        elif isinstance(item, cv.Data):
            label = item.label
            if label is None:
                num = tree.topLevelItemCount()
                label = "Data %i" % num
            parent = tree
        else:
            raise TypeError("Item is not data or subset: %s" % type(item))

        branch = QTreeWidgetItem(parent, [label, '', '', ''])
        branch.setCheckState(0, Qt.Checked)
        self.layer_dict[item] = branch
        self.layer_dict[branch] = item

        tree.expandItem(branch)
        self.sync_layer(item)

    def sync_layer(self, item):
        """ Syncs the content in the desired row of the layer tree,
        based on the state information of the object it refers to """
        if item not in self.layer_dict:
            return
        style = item.style
        row = self.layer_dict[item]
        pm = QPixmap(20, 20)
        pm.fill(mpl_to_qt4_color(style.color))
        row.setIcon(1, QIcon(pm))
        size = style.markersize
        marker = style.marker
        row.setText(2, marker)
        row.setText(3, '%i' % size)
        ncol = self.ui.layerTree.columnCount()
        [self.ui.layerTree.resizeColumnToContents(i) for i in range(ncol)]

    def populate_layer_tree(self):
        for d in self.client.data:
            print 'adding d'
            self.add_tree_item(d)
            print len(d.subsets)
            for s in d.subsets:
                print 'adding s'
                self.add_tree_item(s)


    def init_layer(self, layer):
        print 'init layer'
        self.add_tree_item(layer)

    def remove_layer(self, layer):
        self.layerTree.removeItemWidget(self.layer_dict[layer])

    def toggle_layer_visibility(self):
        item = self.ui.layerTree.currentItem()
        if item is None: return
        layer = self.layer_dict[item]
        self.client.set_visible(layer, item.checkState(0) == Qt.Checked)

    def dummy(self, *args):
        print "Unimplemented slot"
        print args


if __name__ == "__main__":
    from PyQt4.QtGui import QApplication
    import sys
    import cloudviz as cv
    from cloudviz_toolbar import CloudvizToolbar
    from messagewidget import MessageWidget

    app = QApplication(sys.argv)
    win = QMainWindow()

    data, data2, s, s2 = cv.example_data.pipe()
    dc = cv.DataCollection([data, data2])
    scatter_client = ScatterWidget(dc)
    message_client = MessageWidget()
    tb = CloudvizToolbar(dc,
                         scatter_client.ui.mplWidget.canvas,
                         frame = scatter_client)

    hub = cv.Hub(data, data2, dc, s, s2, scatter_client, message_client, tb)

    win.setCentralWidget(scatter_client)
    win.addToolBar(tb)

    win.show()
    message_client.show()
    sys.exit(app.exec_())

