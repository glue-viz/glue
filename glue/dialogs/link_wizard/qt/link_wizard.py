from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui
from glue.core.autolinking import find_possible_links

__all__ = ['run_link_wizard']

DESCRIPTION = ("The link wizard '{0}' has identified {1} links, which are "
               "represented by lines in the following visualization. Hover "
               "over the lines to see more details about the links, then "
               "decide whether or not to proceed.")


class LinkWizardPreview(QtWidgets.QDialog):

    def __init__(self, wizard_name, data_collection, links, parent=None):

        super(LinkWizardPreview, self).__init__(parent=parent)

        self._data_collection = data_collection

        self._ui = load_ui('link_wizard.ui', self,
                           directory=os.path.dirname(__file__))

        self._links = links

        self._ui.graph_widget.set_data_collection(data_collection, links)
        self._ui.graph_widget.selection_changed.connect(self._on_data_change_graph)

        self._ui.current_links.setColumnWidth(0, 200)
        self._ui.current_links.setColumnWidth(1, 300)

        self._ui.label.setText(DESCRIPTION.format(wizard_name, len(links)))

        self._ui.button_apply.clicked.connect(self.accept)
        self._ui.button_ignore.clicked.connect(self.reject)

    @avoid_circular
    def _on_data_change_graph(self):
        self._update_links_list()

    def _update_links_list(self):
        self._ui.current_links.clear()
        data1 = getattr(self._ui.graph_widget.selected_node1, 'data', None)
        data2 = getattr(self._ui.graph_widget.selected_node2, 'data', None)
        for link in self._links:
            to_id = link.get_to_id()
            if to_id.parent in (data1, data2):
                for from_id in link.get_from_ids():
                    if from_id.parent in (data1, data2):
                        self._add_link_to_list(link)
                        break

    def _add_link_to_list(self, link):
        current = self._ui.current_links
        from_ids = ', '.join(cid.label for cid in link.get_from_ids())
        to_id = link.get_to_id().label
        item = QtWidgets.QTreeWidgetItem(current.invisibleRootItem(),
                                         [link._using.__name__, from_ids, to_id])
        item.setData(0, Qt.UserRole, link)

    @classmethod
    def suggest_links(cls, wizard_name, data_collection, links):
        widget = cls(wizard_name, data_collection, links)
        apply = widget._ui.exec_()
        if apply:
            data_collection.add_link(links)


def run_link_wizard(data_collection):
    suggestions = find_possible_links(data_collection)
    for wizard_name, links in suggestions.items():
        LinkWizardPreview.suggest_links(wizard_name, data_collection, links)
