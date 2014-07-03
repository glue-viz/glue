"""
Class which embellishes the DataCollectionView with buttons and actions for
editing the data collection
"""

from ...external.qt.QtGui import (QWidget, QMenu,
                                  QAction, QKeySequence, QFileDialog)


from ...external.qt.QtCore import Qt, Signal, QObject

from ..ui.layertree import Ui_LayerTree

from ... import core

from ..link_editor import LinkEditor
from .. import qtutil
from ..qtutil import get_icon, nonpartial
from .custom_component_widget import CustomComponentWidget
from ..actions import act as _act
from ...core.edit_subset_mode import AndMode, OrMode, XorMode, AndNotMode
from .subset_facet import SubsetFacet


@core.decorators.singleton
class Clipboard(object):

    def __init__(self):
        self.contents = None


class LayerAction(QAction):
    _title = ''
    _icon = None
    _tooltip = None
    _enabled_on_init = False
    _shortcut = None
    _shortcut_context = Qt.WidgetShortcut

    def __init__(self, layer_tree_widget):
        self._parent = layer_tree_widget.layerTree
        super(LayerAction, self).__init__(self._title.title(), self._parent)
        self._layer_tree = layer_tree_widget
        if self._icon:
            self.setIcon(get_icon(self._icon))
        if self._tooltip:
            self.setToolTip(self._tooltip)
        self.setEnabled(self._enabled_on_init)
        if self._shortcut_context is not None:
            self.setShortcutContext(self._shortcut_context)
        if self._shortcut:
            self.setShortcut(self._shortcut)
        self._parent.addAction(self)
        self._connect()

    def _connect(self):
        self._parent.selection_changed.connect(
            self.update_enabled)
        self.triggered.connect(nonpartial(self._do_action))

    def selected_layers(self):
        return self._layer_tree.selected_layers()

    @property
    def data_collection(self):
        return self._layer_tree.data_collection

    def update_enabled(self):
        enabled = self._can_trigger()
        self.setEnabled(enabled)
        self.setVisible(enabled)

    def single_selection(self):
        return len(self.selected_layers()) == 1

    def single_selection_subset(self):
        layers = self.selected_layers()
        if len(layers) != 1:
            return False
        return isinstance(layers[0], core.Subset)

    def single_selection_subset_group(self):
        layers = self.selected_layers()
        if len(layers) != 1:
            return False
        return isinstance(layers[0], core.SubsetGroup)

    def _can_trigger(self):
        raise NotImplementedError

    def _do_action(self):
        raise NotImplementedError


class PlotAction(LayerAction):

    """Visualize the selection. Requires GlueApplication"""
    _title = "Plot Data"
    _tooltip = "Make a plot of this selection"

    def __init__(self, tree, app):
        super(PlotAction, self).__init__(tree)
        self.app = app

    def _can_trigger(self):
        if not self.single_selection():
            return False
        return isinstance(self.selected_layers()[0], (core.Subset, core.Data))

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0].data
        self.app.choose_new_data_viewer(data)


class FacetAction(LayerAction):

    """Add a sequence of subsets which facet a ComponentID"""
    _title = "Create faceted subsets"
    _tooltip = "Create faceted subsets"

    def _can_trigger(self):
        return len(self._layer_tree.data_collection) > 0

    def _do_action(self):
        layers = self.selected_layers()
        try:
            default = layers[0].data
        except (AttributeError, TypeError):
            default = None
        SubsetFacet.facet(self._layer_tree.data_collection,
                          parent=self._layer_tree, default=default)


class NewAction(LayerAction):
    _title = "New Subset"
    _tooltip = "Create a new subset"
    _icon = "glue_subset"
    _shortcut = QKeySequence('Ctrl+Shift+N')

    def _can_trigger(self):
        return len(self.data_collection) > 0

    def _do_action(self):
        assert self._can_trigger()
        self.data_collection.new_subset_group()


class ClearAction(LayerAction):
    _title = "Clear subset"
    _tooltip = "Clear current subset"
    _shortcut = QKeySequence('Ctrl+K')

    def _can_trigger(self):
        return self.single_selection_subset_group()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        subset.subset_state = core.subset.SubsetState()


class DeleteAction(LayerAction):
    _title = "Delete Layer"
    _tooltip = "Delete the selected data and/or subset Groups"
    _shortcut = QKeySequence(Qt.Key_Backspace)

    def _can_trigger(self):
        selection = self.selected_layers()
        return all(isinstance(s, (core.Data, core.SubsetGroup))
                   for s in selection)

    def _do_action(self):
        assert self._can_trigger()
        selection = self.selected_layers()
        for s in selection:
            if isinstance(s, core.Data):
                self._layer_tree.data_collection.remove(s)
            else:
                assert isinstance(s, core.SubsetGroup)
                self._layer_tree.data_collection.remove_subset_group(s)


class LinkAction(LayerAction):
    _title = "Link Data"
    _tooltip = "Define links between data sets"
    _data_link_message = "Define links between data sets"
    _icon = "glue_link"

    def __init__(self, *args, **kwargs):
        super(LinkAction, self).__init__(*args, **kwargs)
        self._link_icon = get_icon(self._icon)
        self._unlink_icon = get_icon('glue_unlink')

    def _can_trigger(self):
        return len(self.data_collection) > 0

    def _do_action(self):
        LinkEditor.update_links(self.data_collection)


class SaveAction(LayerAction):
    _title = "Save subset"
    _tooltip = "Save the mask for this subset to a file"

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        save_subset(subset)


class CopyAction(LayerAction):
    _title = "Copy subset"
    _tooltip = "Copy the definition for the selected subset"
    _shortcut = QKeySequence.Copy

    def _can_trigger(self):
        return self.single_selection_subset_group()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        Clipboard().contents = subset.subset_state


class PasteAction(LayerAction):
    _title = "Paste subset"
    _tooltip = "Overwrite selected subset with contents from clipboard"
    _shortcut = QKeySequence.Paste

    def _can_trigger(self):
        if not self.single_selection_subset_group():
            return False
        cnt = Clipboard().contents
        if not isinstance(cnt, core.subset.SubsetState):
            return False
        return True

    def _do_action(self):
        assert self._can_trigger()
        layer = self.selected_layers()[0]
        layer.paste(Clipboard().contents)


class PasteSpecialAction(PasteAction):
    _title = "Paste Special..."
    _tooltip = "Paste with boolean logic"
    _shortcut = None

    def __init__(self, *args, **kwargs):
        super(PasteSpecialAction, self).__init__(*args, **kwargs)
        self.setMenu(self.menu())

    def menu(self):
        m = QMenu()

        a = QAction("Or", m)
        a.setIcon(get_icon('glue_or'))
        a.triggered.connect(nonpartial(self._paste, OrMode))
        m.addAction(a)

        a = QAction("And", m)
        a.setIcon(get_icon('glue_and'))
        a.triggered.connect(nonpartial(self._paste, AndMode))
        m.addAction(a)

        a = QAction("XOR", m)
        a.setIcon(get_icon('glue_xor'))
        a.triggered.connect(nonpartial(self._paste, XorMode))
        m.addAction(a)

        a = QAction("Not", m)
        a.setIcon(get_icon('glue_andnot'))
        a.triggered.connect(nonpartial(self._paste, AndNotMode))
        m.addAction(a)
        return m

    def _paste(self, mode):
        if not self._can_trigger():
            return
        assert self._can_trigger()
        layer = self.selected_layers()[0]
        mode(layer, Clipboard().contents)

    def _do_action(self):
        pass


class Inverter(LayerAction):
    _title = "Invert"
    _icon = "glue_not"
    _tooltip = "Invert selected subset"

    def _can_trigger(self):
        """ Can trigger iff one subset is selected """
        return self.single_selection_subset_group()

    def _do_action(self):
        """Replace selected subset with its inverse"""
        assert self._can_trigger()
        subset, = self.selected_layers()
        subset.subset_state = core.subset.InvertState(subset.subset_state)


class MergeAction(LayerAction):
    _title = "Merge datasets"
    _tooltip = "Merge the selected datasets into a single dataset"

    def _can_trigger(self):
        layers = self.selected_layers()
        if len(layers) < 2:
            return False

        if not all(isinstance(l, core.Data) for l in layers):
            return False

        shp = layers[0].shape
        return all(d.shape == shp for d in layers[1:])

    def _do_action(self):
        self.data_collection.merge(*self.selected_layers())


class LayerCommunicator(QObject):
    layer_check_changed = Signal(object, bool)


class LayerTreeWidget(QWidget, Ui_LayerTree):

    """The layertree widget provides a way to visualize the various
    data and subset layers in a Glue session.

    This widget relies on sending/receiving messages to/from the hub
    to maintin synchronization with the data collection it manages. If
    it isn't attached to a hub, interactions may not propagate properly.
    """

    def __init__(self, parent=None):
        Ui_LayerTree.__init__(self)
        QWidget.__init__(self, parent)

        self._signals = LayerCommunicator()
        self._is_checkable = True
        self._layer_check_changed = self._signals.layer_check_changed
        self._layer_dict = {}

        self._actions = {}

        self.setupUi(self)
        self._create_actions()
        self._connect()
        self._data_collection = None
        self._hub = None
        self.layerTree.setDragEnabled(True)

    @property
    def data_collection(self):
        return self._data_collection

    def setup(self, collection):
        self._data_collection = collection
        self._hub = collection.hub
        self.layerTree.set_data_collection(collection)

    def unregister(self, hub):
        """Unsubscribe from hub"""
        self.layerTree.unregister(hub)

    def is_checkable(self):
        """ Return whether checkboxes appear next o layers"""
        return self.layerTree.checkable

    def set_checkable(self, state):
        """ Setw hether checkboxes appear next o layers"""
        self.layerTree.checkable = state

    def selected_layers(self):
        """ Return a list of selected layers (subsets and data objects) """
        return self.layerTree.selected_layers()

    def current_layer(self):
        """Return the layer if a single item is selected, else None """
        layers = self.selected_layers()
        if len(layers) == 1:
            return layers[0]

    def actions(self):
        """ Return the list of actions attached to this widget """
        return self.layerTree.actions()

    def _connect(self):
        """ Connect widget signals to methods """
        self._actions['link'] = LinkAction(self)
        self.layerAddButton.clicked.connect(self._load_data)
        self.layerRemoveButton.clicked.connect(self._actions['delete'].trigger)
        self.linkButton.set_action(self._actions['link'])
        self.newSubsetButton.set_action(self._actions['new'], text=False)

        rbut = self.layerRemoveButton

        def update_enabled():
            return rbut.setEnabled(self._actions['delete'].isEnabled())
        self.layerTree.selection_changed.connect(update_enabled)

    def bind_selection_to_edit_subset(self):
        self.layerTree.selection_changed.connect(
            self._update_editable_subset)

    def _update_editable_subset(self):
        """Update edit subsets to match current selection"""
        layers = self.selected_layers()
        layers.extend(s for l in layers
                      if isinstance(l, core.SubsetGroup)
                      for s in l.subsets)

        for data in self.data_collection:
            data.edit_subset = [s for s in data.subsets if s in layers]

    def _create_component(self):
        CustomComponentWidget.create_component(self.data_collection)

    def _create_actions(self):
        tree = self.layerTree

        sep = QAction("", tree)
        sep.setSeparator(True)
        tree.addAction(sep)

        self._actions['save'] = SaveAction(self)
        self._actions['copy'] = CopyAction(self)
        self._actions['paste'] = PasteAction(self)
        self._actions['paste_special'] = PasteSpecialAction(self)
        self._actions['invert'] = Inverter(self)
        self._actions['new'] = NewAction(self)
        self._actions['clear'] = ClearAction(self)
        self._actions['delete'] = DeleteAction(self)
        self._actions['facet'] = FacetAction(self)
        self._actions['merge'] = MergeAction(self)

        # new component definer
        separator = QAction("sep", tree)
        separator.setSeparator(True)
        tree.addAction(separator)

        a = _act("Define new component", self,
                 tip="Define a new component using python expressions")
        tree.addAction(a)
        a.triggered.connect(nonpartial(self._create_component))
        self._actions['new_component'] = a

        # right click pulls up menu
        tree.setContextMenuPolicy(Qt.ActionsContextMenu)

    def _on_item_change(self, item, column):
        """emit check_state_changed signal when checkbox clicked"""
        if item is None or item not in self or column != 0:
            return
        is_checked = item.checkState(0) == Qt.Checked
        layer = self[item]
        self._layer_check_changed.emit(layer, is_checked)

    def _load_data(self):
        """ Interactively loads data from a data set. Adds
        as new layer """
        from ..glue_application import GlueApplication

        layers = qtutil.data_wizard()
        GlueApplication.add_datasets(self.data_collection, layers)

    def __getitem__(self, key):
        raise NotImplementedError()
        return self.layerTree[key]

    def __setitem__(self, key, value):
        raise NotImplementedError()
        self.layerTree[key] = value

    def __contains__(self, obj):
        return obj in self.layerTree

    def __len__(self):
        return len(self.layerTree)


def save_subset(subset):
    assert isinstance(subset, core.subset.Subset)
    fname, fltr = QFileDialog.getSaveFileName(caption="Select an output name")
    fname = str(fname)
    if not fname:
        return
    subset.write_mask(fname)
