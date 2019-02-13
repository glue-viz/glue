from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue import core
from glue.utils import nonpartial
from glue.config import link_function, link_helper
from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui
from glue.core.state_objects import State
from glue.external.echo import CallbackProperty, SelectionCallbackProperty, ChoiceSeparator
from glue.core.data_combo_helper import DataCollectionComboHelper
from glue.external.echo.qt import autoconnect_callbacks_to_qt

__all__ = ['LinkEditor']


def get_function_name(info):
    item = info[0]
    if hasattr(item, 'display') and item.display is not None:
        return item.display
    else:
        return item.__name__


class LinkEditorState(State):

    data1 = SelectionCallbackProperty()
    data2 = SelectionCallbackProperty()
    links = SelectionCallbackProperty()
    link_type = SelectionCallbackProperty()

    def __init__(self, data_collection):

        super(LinkEditorState, self).__init__()

        self.data1_helper = DataCollectionComboHelper(self, 'data1', data_collection)
        self.data2_helper = DataCollectionComboHelper(self, 'data2', data_collection)

        self.data_collection = data_collection

        self.add_callback('data1', self.on_data_change)
        self.add_callback('data2', self.on_data_change)

        categories = []
        for function in link_function.members:
            if len(function.output_labels) == 1:
                categories.append(function.category)
        for helper in link_helper.members:
            categories.append(helper.category)
        categories = ['General'] + sorted(set(categories) - set(['General']))

        link_types = []

        for category in categories:
            link_types.append(ChoiceSeparator(category))
            for function in link_function.members:
                if function.category == category and len(function.output_labels) == 1:
                    link_types.append(function)
            for helper in link_helper.members:
                if helper.category == category:
                    link_types.append(helper)

        LinkEditorState.link_type.set_choices(self, link_types)
        LinkEditorState.link_type.set_display_func(self, get_function_name)

    def on_data_change(self, *args):

        if self.data1 is None or self.data2 is None:
            LinkEditorState.links.set_choices(self, [])
            return

        links = []
        for link in self.data_collection.external_links:
            to_ids = link.get_to_ids()
            to_data = [to_id.parent for to_id in to_ids]
            from_data = [from_id.parent for from_id in link.get_from_ids()]
            if ((self.data1 in to_data and self.data2 in from_data) or
                    (self.data1 in from_data and self.data2 in to_data)):
                links.append(LinkWrapper(link=link))

        LinkEditorState.links.set_choices(self, links)


class LinkWrapper(State):
    link = CallbackProperty()


# TODO: make links shallow-copiable so that we avoid changing the real ones in-place
# TODO: make data combos not allow same data to be selected twice

class LinkEditor(QtWidgets.QDialog):

    def __init__(self, data_collection, parent=None):

        super(LinkEditor, self).__init__(parent=parent)

        self._data_collection = data_collection
        self._links = list(data_collection.external_links)

        self.state = LinkEditorState(data_collection)

        self._ui = load_ui('link_editor.ui', self,
                           directory=os.path.dirname(__file__))
        autoconnect_callbacks_to_qt(self.state, self._ui)

        self._ui.graph_widget.set_data_collection(data_collection)
        self._ui.graph_widget.selection_changed.connect(self._on_data_change_graph)

        # self._on_data_change_combo()

    @avoid_circular
    def _on_data_change_graph(self):
        self.state.data1 = getattr(self._ui.graph_widget.selected_node1, 'data', None)
        self.state.data2 = getattr(self._ui.graph_widget.selected_node2, 'data', None)
    #
    # def _simple_links(self):
    #     """Return identity links which connect the highlighted items
    #     in each component selector.
    #
    #     Returns:
    #       A list of :class:`~glue.core.ComponentLink` objects
    #       If items are not selected in the component selectors,
    #       an empty list is returned
    #     """
    #     comps = self._selected_components()
    #     if len(comps) != 2:
    #         return []
    #     assert isinstance(comps[0], core.data.ComponentID), comps[0]
    #     assert isinstance(comps[1], core.data.ComponentID), comps[1]
    #     link1 = core.component_link.ComponentLink([comps[0]], comps[1])
    #     return [link1]
    #
    # def _add_new_link(self):
    #
    #     if not self.advanced:
    #         links = self._simple_links()
    #     else:
    #         links = self._ui.signature_editor.links()
    #         self._ui.signature_editor.clear_inputs()
    #
    #     self._links.extend(links)
    #
    #     self._ui.graph_widget.set_links(self._links)
    #     self._update_links_list()
    #
    # def links(self):
    #     return self._links
    #
    # def _remove_link(self):
    #     if self._ui.current_links.currentItem() is None:
    #         return
    #     for item in self._ui.current_links.selectedItems():
    #         link = item.data(0, Qt.UserRole)
    #         self._links.remove(link)
    #     self._ui.graph_widget.set_links(self._links)
    #     self._update_links_list()
    #
    @classmethod
    def update_links(cls, collection):
        widget = cls(collection)
        isok = widget._ui.exec_()
        if isok:
            collection.set_links(widget._links)
    #
    # def _add_link_to_list(self, link):
    #     current = self._ui.current_links
    #     from_ids = ', '.join(cid.label for cid in link.get_from_ids())
    #     to_ids = ', '.join(cid.label for cid in link.get_to_ids())
    #     item = QtWidgets.QTreeWidgetItem(current.invisibleRootItem(),
    #                                      [str(link), from_ids, to_ids])
    #     item.setData(0, Qt.UserRole, link)


def main():
    import numpy as np
    from glue.utils.qt import get_qapp
    from glue.core import Data, DataCollection

    app = get_qapp()

    dc = DataCollection()

    for i in range(10):
        x = np.array([1, 2, 3])
        d = Data(label='data_{0:02d}'.format(i), x=x, y=x * 2)
        dc.append(d)

    LinkEditor.update_links(dc)


if __name__ == "__main__":
    main()
