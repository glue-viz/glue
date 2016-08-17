"""
Class which embellishes the DataCollectionView with buttons and actions for
editing the data collection
"""

from __future__ import absolute_import, division, print_function

import os

from qtpy import QtCore, QtWidgets, QtGui, compat
from qtpy.QtCore import Qt

from glue.core.edit_subset_mode import AndMode, OrMode, XorMode, AndNotMode
from glue.core.qt.data_collection_model import DataCollectionView
from glue.config import single_subset_action
from glue import core
from glue.dialogs.link_editor.qt import LinkEditor
from glue.icons.qt import get_icon
from glue.app.qt.actions import action
from glue.dialogs.custom_component.qt import CustomComponentWidget
from glue.dialogs.subset_facet.qt import SubsetFacet
from glue.dialogs.data_wizard.qt import data_wizard
from glue.utils import nonpartial
from glue.utils.qt import load_ui


@core.decorators.singleton
class Clipboard(object):

    def __init__(self):
        self.contents = None


class LayerAction(QtWidgets.QAction):
    _title = ''
    _icon = None
    _tooltip = None
    _enabled_on_init = False
    _shortcut = None
    _shortcut_context = Qt.WidgetShortcut

    def __init__(self, layer_tree_widget):
        self._parent = layer_tree_widget.ui.layerTree
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
        self.setIconVisibleInMenu(False)

    def _connect(self):
        self._parent.selection_changed.connect(nonpartial(self.update_enabled))
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
    _title = "Create new viewer"
    _tooltip = "Create a new viewer from this object"

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
    _shortcut = QtGui.QKeySequence('Ctrl+Shift+N')
    _icon = 'glue_subset'

    def _can_trigger(self):
        return len(self.data_collection) > 0

    def _do_action(self):
        assert self._can_trigger()
        self.data_collection.new_subset_group()


class ClearAction(LayerAction):
    _title = "Clear subset"
    _tooltip = "Clear current subset"
    _shortcut = QtGui.QKeySequence('Ctrl+K')

    def _can_trigger(self):
        return self.single_selection_subset_group()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        subset.subset_state = core.subset.SubsetState()


class DeleteAction(LayerAction):
    _title = "Delete Layer"
    _tooltip = "Delete the selected data and/or subset Groups"
    _shortcut = QtGui.QKeySequence(Qt.Key_Backspace)

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
    _icon = 'glue_link'

    def __init__(self, *args, **kwargs):
        super(LinkAction, self).__init__(*args, **kwargs)

    def _can_trigger(self):
        return len(self.data_collection) > 0

    def _do_action(self):
        LinkEditor.update_links(self.data_collection)


class MaskifySubsetAction(LayerAction):
    _title = "Transform subset to pixel mask"
    _tooltip = "Transform a subset to a pixel mask"

    def _can_trigger(self):
        return self.single_selection() and \
            isinstance(self.selected_layers()[0], core.Subset)

    def _do_action(self):
        s = self.selected_layers()[0]
        s.subset_state = s.state_as_mask()


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
    _shortcut = QtGui.QKeySequence.Copy

    def _can_trigger(self):
        return self.single_selection_subset_group()

    def _do_action(self):
        assert self._can_trigger()
        subset = self.selected_layers()[0]
        Clipboard().contents = subset.subset_state


class PasteAction(LayerAction):
    _title = "Paste subset"
    _tooltip = "Overwrite selected subset with contents from clipboard"
    _shortcut = QtGui.QKeySequence.Paste

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
        m = QtWidgets.QMenu()

        a = QtWidgets.QAction("Or", m)
        a.setIcon(get_icon('glue_or'))
        a.triggered.connect(nonpartial(self._paste, OrMode))
        m.addAction(a)

        a = QtWidgets.QAction("And", m)
        a.setIcon(get_icon('glue_and'))
        a.triggered.connect(nonpartial(self._paste, AndMode))
        m.addAction(a)

        a = QtWidgets.QAction("XOR", m)
        a.setIcon(get_icon('glue_xor'))
        a.triggered.connect(nonpartial(self._paste, XorMode))
        m.addAction(a)

        a = QtWidgets.QAction("Not", m)
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
    _title = "Invert Subset"
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


class UserAction(LayerAction):

    def __init__(self, layer_tree_widget, callback, **kwargs):
        self._title = kwargs.get('name', 'User Action')
        self._tooltip = kwargs.get('tooltip', None)
        self._icon = kwargs.get('icon', None)
        self._callback = callback
        super(UserAction, self).__init__(layer_tree_widget)


class SingleSubsetUserAction(UserAction):

    """
    User-defined callback functions to expose
    when single subsets are selected.

    Users register new actions via the :member:`glue.config.single_subset_action`
    member

    Callback functions are passed the subset and data collection
    """

    def _can_trigger(self):
        return self.single_selection_subset()

    def _do_action(self):
        subset, = self.selected_layers()
        return self._callback(subset, self.data_collection)


class LayerCommunicator(QtCore.QObject):
    layer_check_changed = QtCore.Signal(object, bool)


class LayerTreeWidget(QtWidgets.QMainWindow):

    """The layertree widget provides a way to visualize the various
    data and subset layers in a Glue session.

    This widget relies on sending/receiving messages to/from the hub
    to maintin synchronization with the data collection it manages. If
    it isn't attached to a hub, interactions may not propagate properly.
    """

    def __init__(self, parent=None):

        super(LayerTreeWidget, self).__init__(parent)

        self.ui = load_ui('layer_tree_widget.ui', None, directory=os.path.dirname(__file__))
        self.setCentralWidget(self.ui)
        self.ui.layerAddButton.setIcon(get_icon('glue_open'))
        self.ui.layerRemoveButton.setIcon(get_icon('glue_delete'))

        self._signals = LayerCommunicator()
        self._is_checkable = True
        self._layer_check_changed = self._signals.layer_check_changed
        self._layer_dict = {}

        self._actions = {}

        self._create_actions()
        self._connect()
        self._data_collection = None
        self._hub = None
        self.ui.layerTree.setDragEnabled(True)

    @property
    def data_collection(self):
        return self._data_collection

    def setup(self, collection):
        self._data_collection = collection
        self._hub = collection.hub
        self.ui.layerTree.set_data_collection(collection)

    def unregister(self, hub):
        """Unsubscribe from hub"""
        self.ui.layerTree.unregister(hub)

    def is_checkable(self):
        """ Return whether checkboxes appear next o layers"""
        return self.ui.layerTree.checkable

    def set_checkable(self, state):
        """ Setw hether checkboxes appear next o layers"""
        self.ui.layerTree.checkable = state

    def selected_layers(self):
        """ Return a list of selected layers (subsets and data objects) """
        return self.ui.layerTree.selected_layers()

    def current_layer(self):
        """Return the layer if a single item is selected, else None """
        layers = self.selected_layers()
        if len(layers) == 1:
            return layers[0]

    def actions(self):
        """ Return the list of actions attached to this widget """
        return self.ui.layerTree.actions()

    def _connect(self):
        """ Connect widget signals to methods """
        self._actions['link'] = LinkAction(self)
        self.ui.layerAddButton.clicked.connect(nonpartial(self._load_data))
        self.ui.layerRemoveButton.clicked.connect(self._actions['delete'].trigger)
        self.ui.linkButton.set_action(self._actions['link'])
        self.ui.newSubsetButton.set_action(self._actions['new'], text=False)

        rbut = self.ui.layerRemoveButton

        def update_enabled():
            return rbut.setEnabled(self._actions['delete'].isEnabled())
        self.ui.layerTree.selection_changed.connect(update_enabled)

    def bind_selection_to_edit_subset(self):
        self.ui.layerTree.selection_changed.connect(
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
        dialog = CustomComponentWidget(self.data_collection)
        dialog.exec_()

    def _create_actions(self):
        tree = self.ui.layerTree

        sep = QtWidgets.QAction("", tree)
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
        self._actions['maskify'] = MaskifySubsetAction(self)

        # new component definer
        separator = QtWidgets.QAction("sep", tree)
        separator.setSeparator(True)
        tree.addAction(separator)

        a = action("Define new component", self,
                 tip="Define a new component using python expressions")
        tree.addAction(a)
        a.triggered.connect(nonpartial(self._create_component))
        self._actions['new_component'] = a

        # user-defined layer actions
        for name, callback, tooltip, icon in single_subset_action:
            self._actions[name] = SingleSubsetUserAction(self, callback,
                                                         name=name,
                                                         tooltip=tooltip,
                                                         icon=icon)

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
        from glue.app.qt import GlueApplication

        layers = data_wizard()
        GlueApplication.add_datasets(self.data_collection, layers)

    def __getitem__(self, key):
        raise NotImplementedError()
        return self.ui.layerTree[key]

    def __setitem__(self, key, value):
        raise NotImplementedError()
        self.ui.layerTree[key] = value

    def __contains__(self, obj):
        return obj in self.ui.layerTree

    def __len__(self):
        return len(self.ui.layerTree)


def save_subset(subset):
    assert isinstance(subset, core.subset.Subset)
    fname, fltr = compat.getsavefilename(caption="Select an output name",
                                         filters='FITS mask (*.fits);; Fits mask (*.fits)')
    fname = str(fname)
    if not fname:
        return
    subset.write_mask(fname)

if __name__ == "__main__":
    from glue.core.data_collection import DataCollection
    collection = DataCollection()
    from glue.utils.qt import get_qapp
    app = get_qapp()
    widget = LayerTreeWidget()
    widget.setup(collection)
    widget.show()
    app.exec_()
