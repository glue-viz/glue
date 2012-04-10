""" Class for a layer tree widget """

from PyQt4.QtGui import QWidget, QTreeWidgetItem, QPixmap, QIcon, QInputDialog
from PyQt4.QtGui import QColorDialog

from PyQt4.QtCore import Qt, pyqtSignal
from ui_layertree import Ui_LayerTree
from qtutil import mpl_to_qt4_color, qt4_to_mpl_color, data_wizard

import cloudviz as cv
import cloudviz.message as msg

class LayerTreeWidget(QWidget, Ui_LayerTree, cv.HubListener):
    """The layertree widget provides a way to visualize the various
    data and subset layers in a Glue session.

    This widget relies on sending/receiving messages to/from the hub
    to maintin synchronization with the data collection it manages. If
    it isn't attached to a hub, interactions may not propagate properly.

    """

    def __init__(self, parent=None):
        Ui_LayerTree.__init__(self)
        QWidget.__init__(self, parent)
        cv.HubListener.__init__(self)

        self._layer_dict = {}
        self.setupUi(self)
        self.setup_drag_drop()
        self._connect()

        self._data_collection = cv.DataCollection()

    def setup_drag_drop(self):
        self.layerTree.setDragEnabled(True)

    def current_layer(self):
        return self[self.layerTree.currentItem()]

    @property
    def data_collection(self):
        return self._data_collection

    @data_collection.setter
    def data_collection(self, collection):
        """ Define which data collection this widget manages

        Inputs
        ------
        collection : DataCollection instance to manage
        """
        self.remove_all_layers()

        self._data_collection = collection
        for data in self._data_collection:
            self.add_layer(data)
            for subset in data.subsets:
                self.add_layer(subset)

    def remove_all_layers(self):
        """ Remove all layers from widget """
        for data in self._data_collection:
            self.remove_layer(data)

    def _connect(self):
        """ Connect widget signals to methods """
        self.layerTree.itemDoubleClicked.connect(self.edit_current_layer)
        self.layerAddButton.pressed.connect(self.load_data)
        self.layerRemoveButton.pressed.connect(self.remove_and_delete_selected)
        tree = self.layerTree
        rbut = self.layerRemoveButton
        can_remove = lambda: rbut.setEnabled(tree.currentItem() is not None and
                                             tree.currentItem().isSelected())
        self.layerTree.itemSelectionChanged.connect(can_remove)

    def remove_and_delete_selected(self):
        """ Remove the currently selected layer from layer
        tree. Disconnects layer from session, removing it on all other
        clients. """
        widget_item = self.layerTree.currentItem()
        if not widget_item or not widget_item.isSelected():
            return

        layer = self[widget_item]
        self.remove_layer(layer)

        if isinstance(layer, cv.Data):
            self._data_collection.remove(layer)
        else:
            assert isinstance(layer, cv.Subset)
            layer.unregister()

    def _edit_layer_color(self, layer):
        """ Interactively edit a layer's color """
        dialog = QColorDialog()
        initial = mpl_to_qt4_color(layer.style.color)
        color = dialog.getColor(initial = initial)
        layer.style.color = qt4_to_mpl_color(color)

    def _edit_layer_symbol(self, layer):
        """ Interactively edit a layer's symbol """
        dialog = QInputDialog()
        symb, isok = dialog.getItem(None, 'Pick a Symbol',
                                  'Pick a Symbol',
                                  ['.', 'o', 'v', '>', '<', '^'])
        if isok:
            layer.style.marker = symb

    def _edit_layer_point_size(self, layer):
        """ Interactively edit a layer's point size """
        dialog = QInputDialog()
        size, isok = dialog.getInt(None, 'Point Size', 'Point Size',
                                 value = layer.style.markersize,
                                 min = 1, max = 1000, step = 1)
        if isok:
            layer.style.markersize = size

    def _edit_layer_label(self, layer):
        """ Interactively edit a layer's label """
        dialog = QInputDialog()
        label, isok = dialog.getText(None, 'New Label:', 'New Label:')
        if isok:
            layer.style.label = str(label)

    def edit_current_layer(self):
        """ Allow user to interactively set layer properties of
        the currently highlighted layer """

        column = self.layerTree.currentColumn()
        item = self.layerTree.currentItem()
        if item is None:
            return

        layer = self[item]

        if column == 0:
            self._edit_layer_label(layer)
        if column == 1:
            self._edit_layer_color(layer)
        elif column == 2:
            self._edit_layer_symbol(layer)
        elif column == 3:
            self._edit_layer_point_size(layer)

        # this triggers toggling of expansion state. re-toggle
        expand = item.isExpanded()
        item.setExpanded(not expand)

    def register_to_hub(self, hub):
        """ Connect to hub, to automatically maintain
        synchronization with the underlying datacollection object

        hub: Hub instance to register to
        """
        data_filt = lambda x: x.sender.data in self._data_collection
        dc_filt = lambda x: x.sender is self._data_collection

        hub.subscribe(self,
                      msg.SubsetCreateMessage,
                      handler=lambda x:self.add_layer(x.sender),
                      filter = data_filt)
        hub.subscribe(self,
                      msg.SubsetUpdateMessage,
                      handler=lambda x:self.sync_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.SubsetDeleteMessage,
                      handler=lambda x:self.remove_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.DataCollectionAddMessage,
                      handler=lambda x:self.add_layer(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=lambda x:self.remove_layer(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=lambda x:self.sync_layer(x.sender),
                      filter=data_filt)

    def load_data(self):
        """ Interactively loads data from a data set. Adds
        as new layer """

        layer = data_wizard()
        if layer:
            self._data_collection.append(layer)

    def _ensure_in_collection(self, layer):
        """Makes sure a layer's data is in the data collection.
        Adds if needed"""

        data = layer.data
        if data not in self._data_collection:
            self._data_collection.append(data)

    def _ensure_parent_present(self, layer):
        """ If layer is a subset, make sure that the
        parent data object is already in the layer tree.
        Add if necessary """
        if isinstance(layer, cv.Data):
            return
        if layer.data in self:
            return
        self.add_layer(layer.data)

    def add_layer(self, layer):
        """ Add a new object to the layer tree.

        If it is not already, the layer will be added
        to the layer tree's affiliated data collection object

        Parameters
        ----------
        layer : new data or subset object.
        """
        if layer in self:
            return

        tree = self.layerTree
        self._ensure_in_collection(layer)
        self._ensure_parent_present(layer)

        if isinstance(layer, cv.Subset):
            data = layer.data
            assert data in self._data_collection
            parent = self[data]
            label = layer.style.label
            if label is None:
                ct = parent.childCount()
                datanum = tree.indexOfTopLevelItem(parent)
                label = "Subset %i.%i" % (datanum, ct)
                layer.style.label = label
        elif isinstance(layer, cv.Data):
            assert layer in self._data_collection
            parent = tree
            label = layer.style.label
            if label is None:
                num = tree.topLevelItemCount()
                label = "Data %i" % num
                layer.style.label = label
        else:
            raise TypeError("Input not a data or subset: %s" % type(layer))

        branch = QTreeWidgetItem(parent, [label, '', '', ''])
        branch.setCheckState(0, Qt.Checked)

        self[layer] = branch
        self[branch] = layer

        tree.expandItem(branch)
        self.sync_layer(layer)

    def remove_layer(self, layer):
        """ Remove a data or subset from the layer tree.

        Inputs:
        -------
        layer : Subset or Data object to remove
        """
        if layer not in self:
            return

        widget_item = self[layer]
        parent = widget_item.parent()
        if parent:
            parent.removeChild(widget_item)
        else:
            index = self.layerTree.indexOfTopLevelItem(widget_item)
            if index >= 0:
                self.layerTree.takeTopLevelItem(index)

        self.pop(layer)
        self.pop(widget_item)


    def pop(self, key):
        """ Remove a layer or widget item from the
        layer/widget-item mapping dictionary """
        try:
            self._layer_dict.pop(key)
        except KeyError:
            pass

    def sync_layer(self, layer):
        """ Sync columns of display tree, to
        reflect the current style settings of the given layer

        Inputs:
        -------
        layer : data or subset object
        """
        assert layer in self, "Requested layer not in hierarchy: %s" % layer

        style = layer.style
        widget_item = self[layer]
        pixm = QPixmap(20, 20)
        pixm.fill(mpl_to_qt4_color(style.color))
        widget_item.setIcon(1, QIcon(pixm))
        marker = style.marker
        size = style.markersize
        label = style.label

        widget_item.setText(0, label)
        widget_item.setText(2, marker)
        widget_item.setText(3, "%i" % size)
        ncol = self.layerTree.columnCount()
        for i in range(ncol):
            self.layerTree.resizeColumnToContents(i)

    def is_layer_present(self, layer):
        return layer in self and not self[layer].isHidden()

    def __getitem__(self, key):
        return self._layer_dict[key]

    def __setitem__(self, key, value):
        self._layer_dict[key] = value

    def __contains__(self, obj):
        return obj in self._layer_dict


def main():
    """ Display a layer tree """
    from PyQt4.QtGui import QApplication, QMainWindow
    import sys

    hub = cv.Hub()
    app = QApplication(sys.argv)
    win = QMainWindow()
    ltw = LayerTreeWidget()
    ltw.register_to_hub(hub)
    ltw.data_collection.register_to_hub(hub)
    win.setCentralWidget(ltw)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

