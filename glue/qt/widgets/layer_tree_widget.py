"""
Class which embellishes the DataCollectionView with buttons and actions for
editing the data collection
"""
import operator

from ...external.qt.QtGui import (QWidget, QIcon, QMenu,
                                  QAction, QKeySequence, QFileDialog)


from ...external.qt.QtCore import Qt, Signal, QObject

from ..ui.layertree import Ui_LayerTree

from ... import core

from ..link_editor import LinkEditor
from .. import qtutil
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
            self.setIcon(QIcon(self._icon))
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
        self._parent.itemSelectionChanged.connect(
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


class PlotAction(LayerAction):
    """Visualize the selection. Requires GlueApplication"""
    _title = "Plot Data"
    _tooltip = "Make a plot of this selection"

    def __init__(self, tree, app):
        super(PlotAction, self).__init__(tree)
        self.app = app

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0].data
        self.app.new_data_viewer(data)


class FacetAction(LayerAction):
    """Add a sequence of subsets which facet a ComponentID"""
    _title = "Create faceted subsets"
    _tooltip = "Create faceted subsets"

    def _can_trigger(self):
        return len(self._layer_tree.data_collection) > 0

    def _do_action(self):
        layers = self.selected_layers()
        default = layers[0].data if len(layers) > 0 else None
        SubsetFacet.facet(self._layer_tree.data_collection,
                          parent=self._layer_tree, default=default)


class NewAction(LayerAction):
    _title = "New Subset"
    _tooltip = "Create a new subset for the selected data"
    _icon = ":icons/glue_subset.png"
    _shortcut = QKeySequence('Ctrl+Shift+N')

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0].data
        data.new_subset()


class ClearAction(LayerAction):
    _title = "Clear subset"
    _tooltip = "Clear current subset"
    _shortcut = QKeySequence('Ctrl+K')

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        subset.subset_state = core.subset.SubsetState()


class DuplicateAction(LayerAction):
    _title = "Duplicate subset"
    _tooltip = "Duplicate the current subset"
    _shortcut = QKeySequence('Ctrl+D')

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        new = subset.data.new_subset()
        new.paste(subset)


class DeleteAction(LayerAction):
    _title = "Delete Layer"
    _tooltip = "Delete the selected data and/or subsets"
    _shortcut = QKeySequence(Qt.Key_Backspace)

    def _can_trigger(self):
        selection = self.selected_layers()
        return len(selection) > 0

    def _do_action(self):
        assert self._can_trigger()
        selection = self.selected_layers()
        for s in selection:
            if isinstance(s, core.Subset):
                s.delete()
            else:
                self._layer_tree.data_collection.remove(s)


class LinkAction(LayerAction):
    _title = "Link Data"
    _tooltip = "Define links between data sets"
    _data_link_message = "Define links between data sets"
    _icon = ":icons/glue_link.png"

    def __init__(self, *args, **kwargs):
        super(LinkAction, self).__init__(*args, **kwargs)
        self._link_icon = QIcon(self._icon)
        self._unlink_icon = QIcon(":icons/glue_unlink.png")

    def selection_count(self):
        return len(self.selected_layers())

    @property
    def data_collection(self):
        return self._layer_tree.data_collection

    def _can_trigger(self):
        return self.selection_count() > 0

    def _do_action(self):
        LinkEditor.update_links(self.data_collection)


class LoadAction(LayerAction):
    _title = "Load subset"
    _tooltip = "Load a subset from a file"

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        load_subset(subset)


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
        return self.single_selection_subset()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        Clipboard().contents = subset.subset_state


class PasteAction(LayerAction):
    _title = "Paste subset"
    _tooltip = "Overwrite selected subset with contents from clipboard"
    _shortcut = QKeySequence.Paste

    def _can_trigger(self):
        if not self.single_selection_subset():
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
        a.setIcon(QIcon(':/icons/glue_or.png'))
        a.triggered.connect(lambda: self._paste(OrMode))
        m.addAction(a)

        a = QAction("And", m)
        a.setIcon(QIcon(':/icons/glue_and.png'))
        a.triggered.connect(lambda: self._paste(AndMode))
        m.addAction(a)

        a = QAction("XOR", m)
        a.setIcon(QIcon(':/icons/glue_xor.png'))
        a.triggered.connect(lambda: self._paste(XorMode))
        m.addAction(a)

        a = QAction("Not", m)
        a.setIcon(QIcon(':/icons/glue_andnot.png'))
        a.triggered.connect(lambda: self._paste(AndNotMode))
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


class ChangeLabelAction(LayerAction):
    _title = "Change label"
    _tooltip = "Edit layer label"

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        qtutil.edit_layer_label(self.selected_layers()[0])


class ChangeColorAction(LayerAction):
    _title = "Change color"
    _tooltip = "Edit layer color"

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        qtutil.edit_layer_color(self.selected_layers()[0])


class ChangeSizeAction(LayerAction):
    _title = "Change size"
    _tooltip = "Edit symbol size"

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        qtutil.edit_layer_point_size(self.selected_layers()[0])


class ChangeMarkerAction(LayerAction):
    _title = "Change symbol"
    _tooltip = "Edit plot symbol"

    def _can_trigger(self):
        return self.single_selection()

    def _do_action(self):
        qtutil.edit_layer_symbol(self.selected_layers()[0])


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


class ReduceCombiner(LayerAction):
    """Combine subsets via reduce and a binary method"""
    _reduce_func = None

    def _can_trigger(self):
        """ Action can be triggered iff >=2 subsets from same data selected """
        layers = self.selected_layers()
        if len(layers) < 2:
            return False
        if not all(l.data is layers[0].data for l in layers):
            return False
        if not all(isinstance(l, core.Subset) for l in layers):
            return False

        return True

    def _do_action(self):
        """ Repeatedly apply state_class to reduce selections
        """
        assert self._can_trigger()

        layers = self.selected_layers()
        data = layers[0].data
        subset = data.new_subset()
        subset.subset_state = reduce(self._reduce_func,
                                     (l.subset_state for l in layers))


class BinaryCombiner(ReduceCombiner):
    """A reduce combiner that works on exactly 2 subsets"""
    def _can_trigger(self):
        """ Action can be triggered iff 2 subsets from same data selected """
        if not super(BinaryCombiner, self)._can_trigger():
            return False
        return len(self.selected_layers()) == 2


class XorCombiner(BinaryCombiner):
    _title = "XOR Combine"
    _icon = ':icons/glue_xor.png'
    _tooltip = 'Define new subset as non-intersection of selection'
    _reduce_func = operator.xor


class OrCombiner(ReduceCombiner):
    _title = "Union Combine"
    _icon = ':icons/glue_or.png'
    _tooltip = 'Define new subset as union of selection'
    _reduce_func = operator.or_


class AndCombiner(ReduceCombiner):
    _title = "Intersection Combine"
    _icon = ':icons/glue_and.png'
    _tooltip = 'Define new subset as intersection of selection'
    _reduce_func = operator.and_


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

    def setup(self, collection, hub):
        self._data_collection = collection
        self._hub = hub
        self.layerTree.setup(collection, hub)

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
        items = self.layerTree.selectedItems()
        result = [self[item] for item in items]
        return result

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
        self.layerTree.itemDoubleClicked.connect(self.edit_current_layer)
        self.layerAddButton.clicked.connect(self._load_data)
        self.layerRemoveButton.clicked.connect(self._actions['delete'].trigger)
        self.linkButton.set_action(self._actions['link'])
        self.newSubsetButton.set_action(self._actions['new'], text=False)

        rbut = self.layerRemoveButton

        def update_enabled():
            return rbut.setEnabled(self._actions['delete'].isEnabled())
        self.layerTree.itemSelectionChanged.connect(update_enabled)
        self.layerTree.itemChanged.connect(self._on_item_change)

    def bind_selection_to_edit_subset(self):
        self.layerTree.itemSelectionChanged.connect(
            self._update_editable_subset)

    def _update_editable_subset(self):
        """Update edit subsets to match current selection"""
        layers = self.selected_layers()
        for data in self.data_collection:
            data.edit_subset = [s for s in data.subsets if s in layers]

    def _create_component(self):
        CustomComponentWidget.create_component(self.data_collection)

    def _create_actions(self):
        tree = self.layerTree

        sep = QAction("Style", tree)
        sep.setSeparator(True)
        tree.addAction(sep)

        self._actions['label'] = ChangeLabelAction(self)
        self._actions['color'] = ChangeColorAction(self)
        self._actions['size'] = ChangeSizeAction(self)
        self._actions['marker'] = ChangeMarkerAction(self)

        sep = QAction("", tree)
        sep.setSeparator(True)
        tree.addAction(sep)

        self._actions['load'] = LoadAction(self)
        self._actions['save'] = SaveAction(self)
        self._actions['copy'] = CopyAction(self)
        self._actions['paste'] = PasteAction(self)
        self._actions['paste_special'] = PasteSpecialAction(self)
        self._actions['new'] = NewAction(self)
        self._actions['clear'] = ClearAction(self)
        self._actions['duplicate'] = DuplicateAction(self)
        self._actions['delete'] = DeleteAction(self)
        self._actions['facet'] = FacetAction(self)

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

    def _on_item_change(self, item, column):
        """emit check_state_changed signal when checkbox clicked"""
        if item is None or item not in self or column != 0:
            return
        is_checked = item.checkState(0) == Qt.Checked
        layer = self[item]
        self._layer_check_changed.emit(layer, is_checked)

    def edit_current_layer(self):
        """ Allow user to interactively set layer properties of
        the currently highlighted layer """
        item = self.layerTree.currentItem()
        if item is None:
            return

        layer = self[item]
        qtutil.edit_layer_label(layer)

        # this triggers toggling of expansion state. re-toggle
        expand = item.isExpanded()
        item.setExpanded(not expand)

    def _load_data(self):
        """ Interactively loads data from a data set. Adds
        as new layer """
        layers = qtutil.data_wizard()
        for layer in layers:
            self.data_collection.append(layer)

    def is_layer_present(self, layer):
        return layer in self and not self[layer].isHidden()

    def __getitem__(self, key):
        return self.layerTree[key]

    def __setitem__(self, key, value):
        self.layerTree[key] = value

    def __contains__(self, obj):
        return obj in self.layerTree

    def __len__(self):
        return len(self.layerTree)


def load_subset(subset):
    assert isinstance(subset, core.subset.Subset)
    dialog = QFileDialog()
    file_name, fltr = str(dialog.getOpenFileName(caption="Select a subset"))

    if not file_name:
        return

    try:
        subset.read_mask(file_name)
    except Exception as e:
        print "Exception raised -- could not load\n%s" % e


def save_subset(subset):
    assert isinstance(subset, core.subset.Subset)
    dialog = QFileDialog()
    file_name, fltr = str(dialog.getSaveFileName(
        caption="Select an output name"))
    if not file_name:
        return

    subset.write_mask(file_name)
