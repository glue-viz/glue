import os

from qtpy import QtWidgets

from glue.config import link_function, link_helper
from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui
from echo.qt import autoconnect_callbacks_to_qt
from echo.qt.connect import UserDataWrapper, connect_combo_selection
from glue.dialogs.link_editor.state import LinkEditorState

__all__ = ['LinkEditor', 'main']

N_COMBO_MAX = 10


def get_function_name(info):
    item = info[0]
    if hasattr(item, 'display') and item.display is not None:
        return item.display
    else:
        return item.__name__


class LinkMenu(QtWidgets.QMenu):

    def __init__(self, parent=None):

        super(LinkMenu, self).__init__(parent=parent)

        categories = []
        for function in link_function.members:
            if len(function.output_labels) == 1:
                categories.append(function.category)
        for helper in link_helper.members:
            categories.append(helper.category)
        categories = ['General'] + sorted(set(categories) - set(['General']))

        for category in categories:
            submenu = self.addMenu(category)
            for function in link_function.members:
                if function.category == category and len(function.output_labels) == 1:
                    action = submenu.addAction(get_function_name(function))
                    action.setData(UserDataWrapper(function))
            for helper in link_helper.members:
                if helper.category == category:
                    action = submenu.addAction(get_function_name(helper))
                    action.setData(UserDataWrapper(helper))


class LinkEditorWidget(QtWidgets.QWidget):

    def __init__(self, data_collection, suggested_links=None, parent=None):

        super(LinkEditorWidget, self).__init__(parent=parent)

        self._data_collection = data_collection

        self.state = LinkEditorState(data_collection, suggested_links=suggested_links)

        self._connections = []

        self._ui = load_ui('link_editor_widget.ui', self,
                           directory=os.path.dirname(__file__))
        self._handlers = autoconnect_callbacks_to_qt(self.state, self._ui)

        self._set_up_combos()

        self._ui.graph_widget.set_data_collection(data_collection, new_links=self.state.links)
        self._ui.graph_widget.selection_changed.connect(self._on_data_change_graph)

        self._menu = LinkMenu(parent=self._ui.button_add_link)
        self._menu.triggered.connect(self._add_link)
        self._ui.button_add_link.setMenu(self._menu)

        self._watched_links = []

        self.state.add_callback('data1', self._on_data_change)
        self.state.add_callback('data2', self._on_data_change)
        self._on_data_change()

        self.state.add_callback('data1', self._on_data_change_always)
        self.state.add_callback('data2', self._on_data_change_always)
        self._on_data_change_always()

        self.state.add_callback('current_link', self._on_current_link_change)
        self._on_current_link_change()

    def _add_link(self, action):
        self.state.new_link(action.data().data)

    def _set_up_combos(self):

        # Set up combo boxes - for now we hard-code the maximum number, but
        # we could do this more smartly by checking existing links and all
        # possible links in registry to figure out max number needed.

        self.att_names1 = []
        self.att_combos1 = []

        for combo_idx in range(N_COMBO_MAX):
            label_widget = QtWidgets.QLabel()
            combo_widget = QtWidgets.QComboBox(parent=self._ui)
            self.att_names1.append(label_widget)
            self.att_combos1.append(combo_widget)
            self._ui.combos1.addWidget(label_widget, combo_idx, 0)
            self._ui.combos1.addWidget(combo_widget, combo_idx, 1)

        self.att_names2 = []
        self.att_combos2 = []

        for combo_idx in range(N_COMBO_MAX):
            label_widget = QtWidgets.QLabel()
            combo_widget = QtWidgets.QComboBox(parent=self._ui)
            self.att_names2.append(label_widget)
            self.att_combos2.append(combo_widget)
            self._ui.combos2.addWidget(label_widget, combo_idx, 0)
            self._ui.combos2.addWidget(combo_widget, combo_idx, 1)

    @avoid_circular
    def _on_attribute_combo_change(self, *args, **kwargs):
        # Force a re-sync of the choices
        self._handlers['listsel_current_link'].update_widget(self.state.current_link, force=True)

    @avoid_circular
    def _on_data_change_graph(self):
        self.state.data1 = getattr(self._ui.graph_widget.selected_node1, 'data', None)
        self.state.data2 = getattr(self._ui.graph_widget.selected_node2, 'data', None)

    @avoid_circular
    def _on_data_change(self, *args):
        self._ui.graph_widget.manual_select(self.state.data1, self.state.data2)

    def _on_data_change_always(self, *args):
        # This should always run even when the change comes from the graph
        enabled = self.state.data1 is not None and self.state.data2 is not None
        self._ui.button_add_link.setEnabled(enabled)
        self._ui.button_simple_link.setEnabled(enabled)

    def _on_current_link_change(self, *args):

        # We update the link details panel on the right

        for connnection in self._connections:
            connnection.disconnect()

        self._connections = []

        link = self.state.current_link

        if link is None:
            self._ui.button_remove_link.setEnabled(False)
            self._ui.link_details.setText('')
            self._ui.combos1_header.hide()
            self._ui.combos2_header.hide()
            for widget in self.att_combos1 + self.att_names1 + self.att_combos2 + self.att_names2:
                widget.hide()
            return

        self._ui.button_remove_link.setEnabled(True)

        self._ui.link_details.setText(link.description)

        if link.data1 is self.state.data1:
            data1_names = link.names1
        else:
            data1_names = link.names2

        for idx, (label, combo) in enumerate(zip(self.att_names1, self.att_combos1)):
            if idx < len(data1_names):
                combo.show()
                label.show()
                label.setText(data1_names[idx])
                disconnector = connect_combo_selection(link, data1_names[idx], combo)
                self._connections.append(disconnector)
            else:
                label.hide()
                combo.hide()

        if link.data1 is self.state.data2:
            data2_names = link.names1
        else:
            data2_names = link.names2

        for idx, (label, combo) in enumerate(zip(self.att_names2, self.att_combos2)):
            if idx < len(data2_names):
                combo.show()
                label.show()
                label.setText(data2_names[idx])
                disconnector = connect_combo_selection(link, data2_names[idx], combo)
                self._connections.append(disconnector)
            else:
                label.hide()
                combo.hide()

        # Headers aren't needed if data2_names is 0 (legacy mode for old link
        # helpers where all attributes are 'inputs')
        if len(data2_names) == 0:
            self._ui.combos1_header.hide()
            self._ui.combos2_header.hide()
        else:
            self._ui.combos1_header.show()
            self._ui.combos2_header.show()

        self._ui.graph_widget.set_links(self.state.links)

        # When the user changes one of the attributes, we need to update the
        # main list of links to show the attributes being linked. This is
        # actually tricker than it sounds, and to solve this we listen for
        # changes in any of the link properties. We don't need to unsubscribe
        # since there's no harm in keeping them connected, and we only need to
        # subscribe to links that have been shown at least once as the other
        # ones can't be changed by users.
        if link not in self._watched_links:
            link.add_global_callback(self._on_attribute_combo_change)
            self._watched_links.append(link)


class LinkEditor(QtWidgets.QDialog):

    def __init__(self, data_collection, suggested_links=None, parent=None):

        super(LinkEditor, self).__init__(parent=parent)

        self._ui = load_ui('link_editor_dialog.ui', self,
                           directory=os.path.dirname(__file__))

        self.link_widget = LinkEditorWidget(data_collection,
                                              suggested_links=suggested_links,
                                              parent=self)

        self._ui.layout().insertWidget(1, self.link_widget)

    def accept(self, *args):
        self.link_widget.state.update_links_in_collection()
        super(LinkEditor, self).accept(*args)

    @classmethod
    def update_links(cls, collection, suggested_links=None):
        widget = cls(collection, suggested_links=suggested_links)
        widget._ui.exec_()


def main():  # pragma: no cover
    import numpy as np
    from glue.main import load_plugins
    from glue.utils.qt import get_qapp
    from glue.core import Data, DataCollection

    load_plugins()

    app = get_qapp()

    dc = DataCollection()

    for i in range(10):
        x = np.array([1, 2, 3])
        d = Data(label='data_{0:02d}'.format(i), x=x, y=x * 2)
        dc.append(d)

    LinkEditor.update_links(dc)
