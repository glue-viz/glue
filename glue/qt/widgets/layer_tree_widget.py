""" Class for a layer tree widget """

from PyQt4.QtGui import QWidget, QTreeWidgetItem, QPixmap, QIcon
from PyQt4.QtGui import QAction, QKeySequence, QFileDialog

from PyQt4.QtCore import Qt, pyqtSignal, QObject

from ..ui.layertree import Ui_LayerTree

from ... import core

from ..link_editor import LinkEditor
from .. import qtutil
from .custom_component_widget import CustomComponentWidget
from ..actions import act as _act

@core.decorators.singleton
class Clipboard(object):
    def __init__(self):
        self.contents = None


class LayerAction(QAction):
    _title = None
    _icon = None
    _tooltip = None
    _enabled_on_init = False
    _shortcut = None
    _shortcut_context = Qt.WidgetShortcut

    def __init__(self, layer_tree_widget):
        parent = layer_tree_widget.layerTree
        super(LayerAction, self).__init__(self._title, parent)
        self._layer_tree = layer_tree_widget
        if self._icon:
            self.setIcon(QIcon(self._icon))
        if self._tooltip:
            self.setToolTip(self._tooltip)
        self.setEnabled(self._enabled_on_init)
        if self._shortcut_context is not None:
            self.setShortcutContext(self._shortcut_context)
        if self._shortcut:
            self.setShortcut(self._shortcut)
        parent.addAction(self)
        self._connect()

    def _connect(self):
        self.parentWidget().itemSelectionChanged.connect(
            self.update_enabled)
        self.triggered.connect(self._do_action)

    def selected_layers(self):
        return self._layer_tree.selected_layers()

    def update_enabled(self):
        self.setEnabled(self._can_trigger())

    def single_selection(self):
        return len(self.selected_layers()) == 1

    def single_selection_subset(self):
        layers = self.selected_layers()
        if len(layers) != 1:
            return False
        return isinstance(layers[0], core.Subset)

    def _can_trigger(self):
        raise NotImplementedError

    def _do_action(self):
        raise NotImplementedError

class NewAction(LayerAction):
    _title = "New Subset"
    _tool_tip = "Create a new subset for the selected data"
    _shortcut = 'Ctrl+Shift+N'

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0].data
        data.new_subset()

class ClearAction(LayerAction):
    _title = "Clear subset"
    _tool_tip = "Clear current subset"
    _shortcut = 'Ctrl+K'

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        subset.subset_state = core.subset.SubsetState()

class DuplicateAction(LayerAction):
    _title = "Duplicate subset"
    _tool_tip = "Duplicate the current subset"
    _shortcut = 'Ctrl+D'

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        new = subset.data.new_subset()
        new.paste(subset)


class DeleteAction(LayerAction):
    _title = "Delete Selection"
    _tool_tip = "Remove the selection"
    _shortcut = QKeySequence.Delete

    def _can_trigger(self):
        selection = self.selected_layers()
        if len(selection) != 1:
            return False
        if selection[0] is selection[0].data.edit_subset:
            return False
        return True

    def _do_action(self):
        #XXX refactor into pure function
        self._layer_tree.remove_and_delete_selected()

class LoadAction(LayerAction):
    _title = "Load subset"
    _tool_tip = "Load a subset from a file"

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        load_subset(subset)


class SaveAction(LayerAction):
    _title = "Save subset"
    _tool_tip = "Save the mask for this subset to a file"

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        save_subset(subset)


class CopyAction(LayerAction):
    _title = "Copy subset"
    _tool_tip = "Copy the definition for the selected subset"
    _shortcut = QKeySequence.Copy

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        Clipboard().contents = subset

class PasteAction(LayerAction):
    _title = "Paste subset"
    _tool_tip = "Overwite selected subset with contents from clipboard"
    _shortcut = QKeySequence.Paste

    def _can_trigger(self):
        if not self.single_selection_subset():
            return False
        cnt = Clipboard().contents
        if not isinstance(cnt, core.subset.Subset):
            return False
        return True

    def _do_action(self):
        assert self._can_trigger()
        layer = self.selected_layers()[0]
        layer.paste(Clipboard().contents)



class Inverter(LayerAction):
    _title = "Invert"
    _icon = ":icons/glue_not.png"
    _tooltip = "Invert selected subset"

    def _can_trigger(self):
        """ Can trigger iff one subset is selected """
        layers = self.selected_layers()
        if len(layers) != 1:
            return False
        if not isinstance(layers[0], core.Subset):
            return False
        return True

    def _do_action(self):
        """Replace selected subset with its inverse"""
        assert self._can_trigger()
        subset, = self.selected_layers()
        subset.subset_state = core.subset.InvertState(subset.subset_state)


class BinaryCombiner(LayerAction):
    _state_class = None

    def _can_trigger(self):
        """ Action can be triggered iff 2 subsets from same data selected """
        layers = self.selected_layers()
        if len(layers) != 2:
            return False
        if layers[0].data is not layers[1].data:
            return False

        if not isinstance(layers[0], core.Subset) or \
           not isinstance(layers[1], core.Subset):
            return False

        return True

    def _do_action(self):
        """ Take binary combination of two selected subsets

        combination uses _state_class, which is overridden by subclasses
        """
        assert self._can_trigger()

        layers = self.selected_layers()
        data = layers[0].data
        subset = data.new_subset()
        subset.subset_state = self._state_class(layers[0].subset_state,
                                        layers[1].subset_state)

class OrCombiner(BinaryCombiner):
    _title = "Union Combine"
    _icon = ':icons/glue_or.png'
    _tooltip = 'Define new subset as union of selection'
    _state_class = core.subset.OrState

class AndCombiner(BinaryCombiner):
    _title = "Intersection Combine"
    _icon = ':icons/glue_and.png'
    _tooltip = 'Define new subset as intersection of selection'
    _state_class = core.subset.AndState

class XorCombiner(BinaryCombiner):
    _title = "XOR Combine"
    _icon = ':icons/glue_xor.png'
    _tooltip = 'Define new subset as non-intersection of selection'
    _state_class = core.subset.XorState

class LayerCommunicator(QObject):
    layer_check_changed = pyqtSignal(object, bool)


class LayerTreeWidget(QWidget, Ui_LayerTree, core.hub.HubListener):
    """The layertree widget provides a way to visualize the various
    data and subset layers in a Glue session.

    This widget relies on sending/receiving messages to/from the hub
    to maintin synchronization with the data collection it manages. If
    it isn't attached to a hub, interactions may not propagate properly.
    """
    def __init__(self, parent=None):
        Ui_LayerTree.__init__(self)
        QWidget.__init__(self, parent)
        core.hub.HubListener.__init__(self)

        self._signals = LayerCommunicator()
        self._is_checkable = True
        self._layer_check_changed = self._signals.layer_check_changed
        self._layer_dict = {}

        self._actions = {}

        self.setupUi(self)
        self._create_actions()
        self._connect()
        self._data_collection = core.data_collection.DataCollection()
        self.layerTree.setDragEnabled(True)

    def is_checkable(self):
        """ Return whether checkboxes appear next o layers"""
        return self._is_checkable

    def set_checkable(self, state):
        """ Setw hether checkboxes appear next o layers"""
        self._is_checkable = state

    def selected_layers(self):
        """ Return a list of selected layers (subsets and data objects) """
        items = self.layerTree.selectedItems()
        result = [self[item] for item in items]
        return result

    def current_layer(self):
        """ Return the selected layer if a single item is selected, else None """
        layers = self.selected_layers()
        if len(layers) == 1:
            return layers[0]

    def _create_component(self):
        CustomComponentWidget.create_component(self.data_collection)

    def actions(self):
        """ Return the list of actions attached to this widget """
        return self.layerTree.actions()

    def _create_actions(self):
        tree = self.layerTree

        self._actions['load'] = LoadAction(self)
        self._actions['save'] = SaveAction(self)
        self._actions['copy'] = CopyAction(self)
        self._actions['paste'] = PasteAction(self)
        self._actions['new'] = NewAction(self)
        self._actions['clear'] = ClearAction(self)
        self._actions['duplicate'] = DuplicateAction(self)
        self._actions['delete'] = DeleteAction(self)

        # combination actions
        separator = QAction("Subset Combination", tree)
        separator.setSeparator(True)
        tree.addAction(separator)

        self._actions['or'] = OrCombiner(self)
        self._actions['and'] = AndCombiner(self)
        self._actions['xor'] = XorCombiner(self)
        self._actions['invert'] = Inverter(self)

        # new component definer
        separator = QAction("sep", tree)
        separator.setSeparator(True)
        tree.addAction(separator)

        a = _act("Define new component", self,
                tip="Define a new component using python expressions")
        tree.addAction(a)
        a.triggered.connect(self._create_component)
        self._actions['new_component'] = a

        # right click pulls up menu
        tree.setContextMenuPolicy(Qt.ActionsContextMenu)

    @property
    def data_collection(self):
        return self._data_collection

    @data_collection.setter
    def data_collection(self, collection):
        """ Define which data collection this widget manages

        :param collection: DataCollection instance to manage
        :type collection: :class:`~glue.core.data_collection.DataCollection`
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
        self.layerRemoveButton.pressed.connect(self._actions['delete'].trigger)
        self.linkButton.pressed.connect(
            lambda: LinkEditor.update_links(self._data_collection))
        tree = self.layerTree
        rbut = self.layerRemoveButton
        def update_enabled():
            return rbut.setEnabled(self._actions['delete'].isEnabled())
        self.layerTree.itemSelectionChanged.connect(update_enabled)
        self.layerTree.itemChanged.connect(self._on_item_change)

    def _on_item_change(self, item, column):
        """emit check_state_changed signal when checkbox clicked"""
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
        self.remove_and_delete_layer(layer)

    def remove_and_delete_layer(self, layer):
        """ Remove a layer from the layer tree, and
        disconnect it from the data collection

        This will generate one of the Delete messages
        """
        if layer is layer.data.edit_subset:
            # do not delete the edit subset
            return
        self.remove_layer(layer)

        if isinstance(layer, core.data.Data):
            self._data_collection.remove(layer)
        else:
            assert isinstance(layer, core.subset.Subset)
            layer.delete()

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
        data_in_dc = lambda x: x.data in self._data_collection

        hub.subscribe(self,
                      core.message.SubsetCreateMessage,
                      handler=lambda x: self.add_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.SubsetUpdateMessage,
                      handler=lambda x: self.sync_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.SubsetDeleteMessage,
                      handler=lambda x: self.remove_layer(x.sender),
                      filter=data_filt)
        hub.subscribe(self,
                      core.message.DataCollectionAddMessage,
                      handler=lambda x: self.add_layer(x.data),
                      filter=dc_filt)
        hub.subscribe(self,
                      core.message.DataCollectionDeleteMessage,
                      handler=lambda x: self.remove_layer(x.data),
                      filter = data_in_dc)
        hub.subscribe(self,
                      core.message.DataUpdateMessage,
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
        if isinstance(layer, core.data.Data):
            return
        if layer.data in self:
            return
        self.add_layer(layer.data)

    def add_layer(self, layer):
        """ Add a new object to the layer tree.

        If it is not already, the layer will be added
        to the layer tree's affiliated data collection object

        :param layer: new data or subset object.
        """
        tree = self.layerTree
        self._ensure_in_collection(layer)
        self._ensure_parent_present(layer)

        if layer in self:
            return

        if isinstance(layer, core.subset.Subset):
            data = layer.data
            assert data in self._data_collection
            parent = self[data]
            label = layer.style.label
            if label is None:
                ct = parent.childCount()
                datanum = tree.indexOfTopLevelItem(parent)
                label = "Subset %i.%i" % (datanum, ct)
                layer.style.label = label
        elif isinstance(layer, core.data.Data):
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

        if isinstance(layer, core.data.Data):
            for subset in layer.subsets:
                self.add_layer(subset)

        self.sync_layer(layer)

    def remove_layer(self, layer):
        """ Remove a data or subset from the layer tree.
        Does not remove from data collection.

        :param layer: subset or data object to remove
        :type layer:
           :class:`~glue.core.subset.Subset` or
           :class:`~glue.core.data.Data`
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

        :param layer: data or subset object
        """
        if layer not in self:
            return

        style = layer.style
        widget_item = self[layer]
        pixm = QPixmap(20, 20)
        pixm.fill(qtutil.mpl_to_qt4_color(style.color))
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
        self.layerTree.set_data(key, value)
        self._layer_dict[key] = value

    def __contains__(self, obj):
        return obj in self._layer_dict

    def __len__(self):
        return len(self._layer_dict)


def load_subset(subset):
    assert isinstance(subset, core.subset.Subset)
    dialog = QFileDialog()
    file_name = str(dialog.getOpenFileName(caption="Select a subset"))

    if not file_name:
        return

    try:
        layer.read_mask(file_name)
    except Exception as e:
        print "Exception raised -- could not load\n%s" % e

def save_subset(subset):
    assert isinstance(subset, core.subset.Subset)
    dialog = QFileDialog()
    file_name = str(dialog.getSaveFileName(
        caption="Select an output name"))
    if not file_name:
        return

    subset.write_mask(file_name)

