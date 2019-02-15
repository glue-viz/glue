from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue import core
from glue.utils import nonpartial
from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui

__all__ = ['LinkEditor']


class LinkEditor(QtWidgets.QDialog):

    def __init__(self, collection, functions=None, parent=None):

        super(LinkEditor, self).__init__(parent=parent)

        self._collection = collection

        self._ui = load_ui('link_editor.ui', self,
                           directory=os.path.dirname(__file__))

        self._links = list(collection.external_links)

        self._ui.graph_widget.set_data_collection(collection)
        self._ui.graph_widget.selection_changed.connect(self._on_data_change_graph)

        self._init_widgets()
        self._connect()

        self._size = None

        if len(self._collection) == 2:
            self._ui.left_components.data = self._collection[0]
            self._ui.right_components.data = self._collection[1]
        else:
            self._ui.left_components.data = None
            self._ui.right_components.data = None

        self._ui.left_components.data_changed.connect(self._on_data_change_combo)
        self._ui.right_components.data_changed.connect(self._on_data_change_combo)

        self._on_data_change_combo()

    @avoid_circular
    def _on_data_change_graph(self):
        self._ui.left_components.data = getattr(self._ui.graph_widget.selected_node1, 'data', None)
        self._ui.right_components.data = getattr(self._ui.graph_widget.selected_node2, 'data', None)
        self._update_links_list()

    @avoid_circular
    def _on_data_change_combo(self):
        graph = self._ui.graph_widget
        graph.manual_select(self._ui.left_components.data, self._ui.right_components.data)
        self._update_links_list()

    def _init_widgets(self):
        self._ui.left_components.setup(self._collection)
        self._ui.right_components.setup(self._collection)
        self._ui.signature_editor.hide()

    def _connect(self):
        self._ui.add_link.clicked.connect(nonpartial(self._add_new_link))
        self._ui.remove_link.clicked.connect(nonpartial(self._remove_link))
        self._ui.toggle_editor.clicked.connect(nonpartial(self._toggle_advanced))

    @property
    def advanced(self):
        return self._ui.signature_editor.isVisible()

    @advanced.setter
    def advanced(self, state):
        """Set whether the widget is in advanced state"""
        self._ui.signature_editor.setVisible(state)
        self._ui.toggle_editor.setText("Basic linking" if state else "Advanced linking")

    def _toggle_advanced(self):
        """Show or hide the signature editor widget"""
        self.advanced = not self.advanced

    def _selected_components(self):
        result = []
        id1 = self._ui.left_components.component
        id2 = self._ui.right_components.component
        if id1:
            result.append(id1)
        if id2:
            result.append(id2)
        return result

    def _simple_links(self):
        """Return identity links which connect the highlighted items
        in each component selector.

        Returns:
          A list of :class:`~glue.core.ComponentLink` objects
          If items are not selected in the component selectors,
          an empty list is returned
        """
        comps = self._selected_components()
        if len(comps) != 2:
            return []
        assert isinstance(comps[0], core.data.ComponentID), comps[0]
        assert isinstance(comps[1], core.data.ComponentID), comps[1]
        link1 = core.component_link.ComponentLink([comps[0]], comps[1])
        return [link1]

    def _add_new_link(self):

        if not self.advanced:
            links = self._simple_links()
        else:
            links = self._ui.signature_editor.links()
            self._ui.signature_editor.clear_inputs()

        self._links.extend(links)

        self._ui.graph_widget.set_links(self._links)
        self._update_links_list()

    def links(self):
        return self._links

    def _remove_link(self):
        if self._ui.current_links.currentItem() is None:
            return
        for item in self._ui.current_links.selectedItems():
            link = item.data(0, Qt.UserRole)
            self._links.remove(link)
        self._ui.graph_widget.set_links(self._links)
        self._update_links_list()

    @classmethod
    def update_links(cls, collection):
        widget = cls(collection)
        isok = widget._ui.exec_()
        if isok:
            collection.set_links(widget._links)

    def _update_links_list(self):
        self._ui.current_links.clear()
        data1 = self._ui.left_components.data
        data2 = self._ui.right_components.data
        if data1 is None or data2 is None:
            return
        links_to_add = set()
        for link in self._links:
            to_ids = link.get_to_ids()
            to_data = [to_id.parent for to_id in to_ids]
            from_data = [from_id.parent for from_id in link.get_from_ids()]
            if (data1 in to_data and data2 in from_data) or (data1 in from_data and data2 in to_data):
                links_to_add.add(link)

        for link in links_to_add:
            self._add_link_to_list(link)

    def _add_link_to_list(self, link):
        current = self._ui.current_links
        from_ids = ', '.join(cid.label for cid in link.get_from_ids())
        to_ids = ', '.join(cid.label for cid in link.get_to_ids())
        item = QtWidgets.QTreeWidgetItem(current.invisibleRootItem(),
                                         [str(link), from_ids, to_ids])
        item.setData(0, Qt.UserRole, link)


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
