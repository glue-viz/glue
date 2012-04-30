""" Class for a layer tree widget """

from PyQt4.QtGui import QWidget, QTreeWidgetItem, QPixmap, QIcon
from PyQt4.QtGui import QAction, QKeySequence, QActionGroup

from PyQt4.QtCore import Qt, pyqtSignal, QObject
from ui_layertree import Ui_LayerTree

import glue
import glue.message as msg
from glue.subset import OrState, AndState, XorState, InvertState
from glue.qt.data_connector import DataConnector
from glue.qt import qtutil
from glue.qt.qtutil import mpl_to_qt4_color


class LayerCommunicator(QObject):
    layer_check_changed = pyqtSignal(object, bool)


class LayerTreeWidget(QWidget, Ui_LayerTree, glue.HubListener):
    """The layertree widget provides a way to visualize the various
    data and subset layers in a Glue session.

    This widget relies on sending/receiving messages to/from the hub
    to maintin synchronization with the data collection it manages. If
    it isn't attached to a hub, interactions may not propagate properly.

    """
    def __init__(self, parent=None):
        Ui_LayerTree.__init__(self)
        QWidget.__init__(self, parent)
        glue.HubListener.__init__(self)

        self._signals = LayerCommunicator()
        self._is_checkable = True
        self._clipboard = None
        self._layer_check_changed = self._signals.layer_check_changed
        self._layer_dict = {}

        self._new_action = None
        self._clear_action = None
        self._duplicate_action = None
        self._copy_action = None
        self._paste_action = None
        self._delete_action = None
        self._or_action = None
        self._xor_action = None
        self._and_action = None
        self._invert_action = None

        self.setupUi(self)
        self.setup_drag_drop()
        self._create_actions()
        self._connect()
        self._data_collection = glue.DataCollection()

    def is_checkable(self):
        return self._is_checkable

    def set_checkable(self, state):
        self._is_checkable = state

    def _combination_actions(self):
        tree = self.layerTree
        group = QActionGroup(tree)

        act = QAction("Union Combine", tree)
        act.setIcon(QIcon(':icons/glue_or.png'))
        act.setEnabled(False)
        act.setToolTip("Define new subset as union of selection")
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._or_combine)
        group.addAction(act)
        tree.addAction(act)
        self._or_action = act

        act = QAction("Intersection Combine", tree)
        act.setIcon(QIcon(':icons/glue_and.png'))
        act.setEnabled(False)
        act.setToolTip("Define new subset as intersection of selection")
        act.triggered.connect(self._and_combine)
        act.setShortcutContext(Qt.WidgetShortcut)
        group.addAction(act)
        tree.addAction(act)
        self._and_action = act

        act = QAction("XOR Combine", tree)
        act.setIcon(QIcon(':icons/glue_xor.png'))
        act.setEnabled(False)
        act.setToolTip("Define new subset as non-intersection of selection")
        act.triggered.connect(self._xor_combine)
        act.setShortcutContext(Qt.WidgetShortcut)
        group.addAction(act)
        tree.addAction(act)
        self._xor_action = act

        act = QAction("Invert", tree)
        act.setIcon(QIcon(':icons/glue_not.png'))
        act.setEnabled(False)
        act.setToolTip("Invert current subset")
        act.triggered.connect(self._invert_subset)
        act.setShortcutContext(Qt.WidgetShortcut)
        group.addAction(act)
        tree.addAction(act)
        self._invert_action = act

    def _invert_subset(self):
        item = self.layerTree.currentItem()
        layer = self[item]
        state = layer.subset_state
        layer.subset_state = InvertState(state)

    def _update_combination_actions_enabled(self):
        state = self._check_can_combine()
        self._or_action.setEnabled(state)
        self._and_action.setEnabled(state)
        self._xor_action.setEnabled(state)

        layer = self.current_layer()
        can_invert = isinstance(layer, glue.Subset)
        self._invert_action.setEnabled(can_invert)

    def _check_can_combine(self, layers=None):
        if layers is None:
            items = self.layerTree.selectedItems()
            layers = [self[i] for i in items]

        if len(layers) != 2:
            return False
        if not isinstance(layers[0], glue.Subset):
            return False
        if not isinstance(layers[1], glue.Subset):
            return False
        if layers[0].data is not layers[1].data:
            return False
        return True

    def _and_combine(self):
        self._binary_combine(AndState)

    def _xor_combine(self):
        self._binary_combine(XorState)

    def _or_combine(self):
        self._binary_combine(OrState)

    def _binary_combine(self, StateClass):
        items = self.layerTree.selectedItems()
        layers = [self[i] for i in items]
        if not self._check_can_combine(layers):
            return
        data = layers[0].data
        new = data.new_subset()
        new.subset_state = StateClass(layers[0].subset_state,
                                      layers[1].subset_state)

    def _create_actions(self):
        tree = self.layerTree

        act = QAction("Copy subset", tree)
        act.setEnabled(False)
        act.setToolTip("Copy the definition for the selected subset")
        act.setShortcut(QKeySequence.Copy)
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._copy_subset)
        tree.addAction(act)
        self._copy_action = act

        act = QAction("Paste subset", tree)
        act.setEnabled(False)
        act.setToolTip("Overwite selected subset with contents from clipboard")
        act.setShortcut(QKeySequence.Paste)
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._paste_subset)
        tree.addAction(act)
        self._paste_action = act

        act = QAction("New subset", tree)
        act.setEnabled(False)
        act.setToolTip("Create a new subset for the selected data")
        act.setShortcut('Ctrl+N')
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._new_subset)
        tree.addAction(act)
        self._new_action = act

        act = QAction("Clear subset", tree)
        act.setEnabled(False)
        act.setToolTip("Clear current selection")
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._clear_subset)
        tree.addAction(act)
        self._clear_action = act

        act = QAction("Duplicate subset", tree)
        act.setEnabled(False)
        act.setToolTip("Duplicate the current subset")
        act.setShortcut('Ctrl+D')
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self._duplicate_subset)
        tree.addAction(act)
        self._duplicate_action = act

        act = QAction("Delete layer", tree)
        act.setEnabled(False)
        act.setToolTip("Remove the highlighted layer")
        act.setShortcut(QKeySequence.Delete)
        act.setShortcutContext(Qt.WidgetShortcut)
        act.triggered.connect(self.remove_and_delete_selected)
        self._delete_action = act
        tree.addAction(act)

        separator = QAction("Subset Combination", tree)
        separator.setSeparator(True)
        tree.addAction(separator)

        self._combination_actions()

        tree.setContextMenuPolicy(Qt.ActionsContextMenu)

    def _clear_subset(self):
        layer = self.current_layer()
        if not isinstance(layer, glue.Subset):
            return
        layer.subset_state = glue.subset.SubsetState()

    def _copy_subset(self):
        layer = self.current_layer()
        assert isinstance(layer, glue.Subset)
        self._clipboard = layer

    def _paste_subset(self):
        layer = self.current_layer()
        assert isinstance(layer, glue.Subset)
        assert self._clipboard is not None
        layer.paste(self._clipboard)

    def _new_subset(self):
        data = self.current_layer().data
        data.new_subset()

    def _duplicate_subset(self):
        layer = self.current_layer()
        subset = layer.data.new_subset()
        subset.paste(layer)

    def _update_actions_enabled(self):
        self._update_combination_actions_enabled()
        layer = self.current_layer()
        has_clipboard = self._clipboard is not None
        single = len(self.layerTree.selectedItems()) == 1

        if single and isinstance(layer, glue.Data):
            self._copy_action.setEnabled(False)
            self._paste_action.setEnabled(False)
            self._new_action.setEnabled(True)
            self._duplicate_action.setEnabled(False)
            self._delete_action.setEnabled(True)
            self._clear_action.setEnabled(False)
        elif single and isinstance(layer, glue.Subset):
            self._copy_action.setEnabled(True)
            self._paste_action.setEnabled(has_clipboard)
            self._new_action.setEnabled(True)
            self._duplicate_action.setEnabled(True)
            self._delete_action.setEnabled(True)
            self._clear_action.setEnabled(True)
        else:
            self._copy_action.setEnabled(False)
            self._paste_action.setEnabled(False)
            self._new_action.setEnabled(False)
            self._duplicate_action.setEnabled(False)
            self._delete_action.setEnabled(False)
            self._clear_action.setEnabled(False)

    def setup_drag_drop(self):
        self.layerTree.setDragEnabled(True)

    def current_layer(self):
        item = self.layerTree.currentItem()
        if item is None:
            return None
        return self[item]

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
        layers = [d for d in self._data_collection]

        for data in layers:
            self.remove_layer(data)

    def _connect(self):
        """ Connect widget signals to methods """
        self.layerTree.itemDoubleClicked.connect(self.edit_current_layer)
        self.layerAddButton.pressed.connect(self._load_data)
        self.layerRemoveButton.pressed.connect(self._delete_action.trigger)
        self.linkButton.pressed.connect(
            lambda: DataConnector.set_connections(self._data_collection)
        )
        tree = self.layerTree
        rbut = self.layerRemoveButton
        can_remove = lambda: rbut.setEnabled(tree.currentItem() is not None and
                                             tree.currentItem().isSelected())
        self.layerTree.itemSelectionChanged.connect(can_remove)
        self.layerTree.itemChanged.connect(self._on_item_change)
        self.layerTree.itemSelectionChanged.connect(
            self._update_actions_enabled)

    def _on_item_change(self, item, column):
        # emit check_state_changed signal when checkbox clicked
        if item is None or item not in self or column != 0:
            return
        is_checked = item.checkState(0) == Qt.Checked
        layer = self[item]
        self._layer_check_changed.emit(layer, is_checked)

    def remove_and_delete_selected(self):
        """ Remove the currently selected layer from layer
        tree. Disconnects layer from session, removing it on all other
        clients. """
        widget_item = self.layerTree.currentItem()
        if not widget_item or not widget_item.isSelected():
            return

        layer = self[widget_item]
        self.remove_layer(layer)

        if isinstance(layer, glue.Data):
            self._data_collection.remove(layer)
        else:
            assert isinstance(layer, glue.Subset)
            layer.unregister()

    def edit_current_layer(self):
        """ Allow user to interactively set layer properties of
        the currently highlighted layer """

        column = self.layerTree.currentColumn()
        item = self.layerTree.currentItem()
        if item is None:
            return

        layer = self[item]

        if column == 0:
            qtutil.edit_layer_label(layer)
        if column == 1:
            qtutil.edit_layer_color(layer)
        elif column == 2:
            qtutil.edit_layer_symbol(layer)
        elif column == 3:
            qtutil.edit_layer_point_size(layer)

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
                      handler=lambda x: self.add_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.SubsetUpdateMessage,
                      handler=lambda x: self.sync_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.SubsetDeleteMessage,
                      handler=lambda x: self.remove_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      msg.DataCollectionAddMessage,
                      handler=lambda x: self.add_layer(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataCollectionDeleteMessage,
                      handler=lambda x: self.remove_layer(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      msg.DataUpdateMessage,
                      handler=lambda x: self.sync_layer(x.sender),
                      filter=data_filt)

    def _load_data(self):
        """ Interactively loads data from a data set. Adds
        as new layer """

        layer = qtutil.data_wizard()
        if layer:
            self.add_layer(layer)

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
        if isinstance(layer, glue.Data):
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
        tree = self.layerTree
        self._ensure_in_collection(layer)
        self._ensure_parent_present(layer)

        if layer in self:
            return

        if isinstance(layer, glue.Subset):
            data = layer.data
            assert data in self._data_collection
            parent = self[data]
            label = layer.style.label
            if label is None:
                ct = parent.childCount()
                datanum = tree.indexOfTopLevelItem(parent)
                label = "Subset %i.%i" % (datanum, ct)
                layer.style.label = label
        elif isinstance(layer, glue.Data):
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
        if self.is_checkable():
            branch.setCheckState(0, Qt.Checked)

        assert layer not in self
        self[layer] = branch
        self[branch] = layer

        tree.expandItem(branch)

        if isinstance(layer, glue.Data):
            for subset in layer.subsets:
                self.add_layer(subset)

        self.sync_layer(layer)

    def remove_layer(self, layer):
        """ Remove a data or subset from the layer tree.
        Does not remove from data collection.

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
        #if layer in self._data_collection:
        #    self._data_collection.remove(layer)

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
        if layer not in self:
            return

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
