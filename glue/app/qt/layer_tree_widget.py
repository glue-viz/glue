"""
Class which embellishes the DataCollectionView with buttons and actions for
editing the data collection
"""

from __future__ import absolute_import, division, print_function

import os
import weakref

from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt

from glue.core.edit_subset_mode import AndMode, OrMode, XorMode, AndNotMode
from glue.config import layer_action
from glue import core
from glue.dialogs.link_editor.qt import LinkEditor
from glue.icons.qt import get_icon
from glue.dialogs.component_arithmetic.qt import ArithmeticEditorWidget
from glue.dialogs.component_manager.qt import ComponentManagerWidget
from glue.dialogs.subset_facet.qt import SubsetFacet
from glue.dialogs.data_wizard.qt import data_wizard
from glue.utils import nonpartial
from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui
from glue.core.message import EditSubsetMessage
from glue.core.hub import HubListener
from glue.app.qt.metadata import MetadataDialog


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
        super(LayerAction, self).__init__(self._title, self._parent)
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

    def single_selection_data(self):
        layers = self.selected_layers()
        if len(layers) != 1:
            return False
        return isinstance(layers[0], core.Data)

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
        self.app = weakref.ref(app)

    def _can_trigger(self):
        if not self.single_selection():
            return False
        return isinstance(self.selected_layers()[0], (core.Subset, core.Data))

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0].data
        app = self.app()
        if app is not None:
            app.choose_new_data_viewer(data)


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
        except (AttributeError, TypeError, IndexError):
            default = None
        SubsetFacet.facet(self._layer_tree.data_collection,
                          parent=self._layer_tree, default=default)


class MetadataAction(LayerAction):

    _title = "View metadata/header"
    _tooltip = "View metadata/header"
    _shortcut = QtGui.QKeySequence('Ctrl+I')

    def _can_trigger(self):
        return self.single_selection_data() or self.single_selection_subset()

    def _do_action(self):
        layers = self.selected_layers()
        if isinstance(layers[0], core.Subset):
            data = layers[0].data
        else:
            data = layers[0]
        MetadataDialog(data).exec_()


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


class ExportDataAction(LayerAction):

    _title = "Export data values"
    _tooltip = "Save the data to a file"

    def _can_trigger(self):
        return self.single_selection_data()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0]
        from glue.core.data_exporters.qt.dialog import export_data
        export_data(data)


class ArithmeticAction(LayerAction):

    _title = "Add/edit arithmetic attributes"
    _tooltip = "Add/edit attributes derived from existing ones"

    def _can_trigger(self):
        return self.single_selection_data()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0]
        print(data.label)
        dialog = ArithmeticEditorWidget(self._layer_tree.data_collection,
                                        initial_data=data)
        dialog.exec_()


class ManageComponentsAction(LayerAction):

    _title = "Reorder/rename data attributes"
    _tooltip = "Reorder/rename data attributes"

    def _can_trigger(self):
        return self.single_selection_data()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0]
        print(data.label)
        dialog = ComponentManagerWidget(self._layer_tree.data_collection,
                                        initial_data=data)
        dialog.exec_()


class ExportSubsetAction(ExportDataAction):

    _title = "Export subset values"
    _tooltip = "Save the data subset to a file"

    def _can_trigger(self):
        return self.single_selection_subset()


class ImportSubsetMaskAction(LayerAction):

    _title = "Import subset mask(s)"
    _tooltip = "Import subset mask from a file"

    def _can_trigger(self):
        return self.single_selection_subset() or self.single_selection_data()

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0]
        from glue.io.qt.subset_mask import QtSubsetMaskImporter
        QtSubsetMaskImporter().run(data, self._layer_tree._data_collection)


class ExportSubsetMaskAction(LayerAction):

    _title = "Export subset mask(s)"
    _tooltip = "Export subset mask to a file"

    def _can_trigger(self):
        return (len(self.data_collection.subset_groups) > 0 and
                (self.single_selection_subset() or self.single_selection_data()))

    def _do_action(self):
        assert self._can_trigger()
        data = self.selected_layers()[0]
        from glue.io.qt.subset_mask import QtSubsetMaskExporter
        QtSubsetMaskExporter().run(data)


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
    """
    User-defined callback functions to expose.

    Users register new actions via the :member:`glue.config.layer_action` member

    Callback functions are passed the layer (or list of layers) and data
    collection
    """

    def __init__(self, layer_tree_widget, callback=None, label='User Action',
                 tooltip=None, icon=None, single=False, data=False,
                 subset_group=False, subset=False):
        self._title = label
        self._tooltip = tooltip
        self._icon = icon
        self._single = single
        self._data = data
        self._subset_group = subset_group
        self._subset = subset
        self._callback = callback
        super(UserAction, self).__init__(layer_tree_widget)

    def _can_trigger(self):
        if self._single:
            if self.single_selection():
                layer = self.selected_layers()[0]
                if isinstance(layer, core.Data) and self._data:
                    return True
                elif isinstance(layer, core.SubsetGroup) and self._subset_group:
                    return True
                elif isinstance(layer, core.Subset) and self._subset:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return len(self.selected_layers()) > 0

    def _do_action(self):
        if self._single:
            subset = self.selected_layers()[0]
            return self._callback(subset, self.data_collection)
        else:
            return self._callback(self.selected_layers(), self.data_collection)


class LayerCommunicator(QtCore.QObject):
    layer_check_changed = QtCore.Signal(object, bool)


class LayerTreeWidget(QtWidgets.QMainWindow, HubListener):

    """The layertree widget provides a way to visualize the various
    data and subset layers in a Glue session.

    This widget relies on sending/receiving messages to/from the hub
    to maintin synchronization with the data collection it manages. If
    it isn't attached to a hub, interactions may not propagate properly.
    """

    def __init__(self, session=None, parent=None):

        super(LayerTreeWidget, self).__init__(parent)

        self.session = session

        self.ui = load_ui('layer_tree_widget.ui', None, directory=os.path.dirname(__file__))
        self.setCentralWidget(self.ui)

        self._signals = LayerCommunicator()
        self._is_checkable = True
        self._layer_check_changed = self._signals.layer_check_changed
        self._layer_dict = {}

        self._actions = {}

        self._create_actions()
        self._data_collection = None
        self._hub = None
        self.ui.layerTree.setDragEnabled(True)
        self.setMinimumSize(300, 0)

    @property
    def data_collection(self):
        return self._data_collection

    def setup(self, collection):
        self._data_collection = collection
        self._hub = collection.hub
        self.ui.layerTree.set_data_collection(collection)
        self.bind_selection_to_edit_subset()

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

    def bind_selection_to_edit_subset(self):
        self.ui.layerTree.selection_changed.connect(self._update_edit_subset)
        self._data_collection.hub.subscribe(self, EditSubsetMessage,
                                            handler=self._update_subset_selection)

    @avoid_circular
    def _update_subset_selection(self, message):
        """
        Update current selection to match edit subsets
        """
        self.ui.layerTree.set_selected_layers(message.subset)

    @avoid_circular
    def _update_edit_subset(self):
        """
        Update edit subsets to match current selection
        """
        mode = self.session.edit_subset_mode
        mode.edit_subset = [s for s in self.selected_layers() if isinstance(s, core.SubsetGroup)]

    def _create_actions(self):
        tree = self.ui.layerTree

        sep = QtWidgets.QAction("", tree)
        sep.setSeparator(True)
        tree.addAction(sep)

        # Actions relating to I/O
        self._actions['save_data'] = ExportDataAction(self)
        self._actions['save_subset'] = ExportSubsetAction(self)
        self._actions['import_subset_mask'] = ImportSubsetMaskAction(self)
        self._actions['export_subset_mask'] = ExportSubsetMaskAction(self)

        self._actions['copy'] = CopyAction(self)
        self._actions['paste'] = PasteAction(self)
        self._actions['paste_special'] = PasteSpecialAction(self)
        self._actions['invert'] = Inverter(self)
        self._actions['new'] = NewAction(self)
        self._actions['clear'] = ClearAction(self)
        self._actions['delete'] = DeleteAction(self)
        self._actions['facet'] = FacetAction(self)
        self._actions['metadata'] = MetadataAction(self)
        self._actions['merge'] = MergeAction(self)
        self._actions['maskify'] = MaskifySubsetAction(self)
        self._actions['link'] = LinkAction(self)
        self._actions['new_component'] = ArithmeticAction(self)
        self._actions['manage_components'] = ManageComponentsAction(self)

        # Add user-defined layer actions. Note that _asdict is actually a public
        # method, but just has an underscore to prevent conflict with
        # namedtuple attributes.
        for item in layer_action:
            self._actions[item.label] = UserAction(self, **item._asdict())

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


if __name__ == "__main__":
    from glue.core.data_collection import DataCollection
    collection = DataCollection()
    from glue.utils.qt import get_qapp
    app = get_qapp()
    widget = LayerTreeWidget()
    widget.setup(collection)
    widget.show()
    app.exec_()
