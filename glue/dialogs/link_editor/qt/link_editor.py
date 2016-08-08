from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets
from glue import core
from glue.utils.qt import load_ui

# FIXME: at the moment we need to make sure that custom widgets are imported,
# otherwise they don't get appended to CUSTOM_QWIDGETS. We need to find a
# better long-term solution.
from glue.dialogs.common.qt.component_selector import ComponentSelector
from glue.core.qt.mime import GlueMimeListWidget
from glue.dialogs.link_editor.qt.link_equation import LinkEquation

__all__ = ['LinkEditor']


class LinkEditor(QtWidgets.QDialog):

    def __init__(self, collection, functions=None, parent=None):

        super(LinkEditor, self).__init__(parent=parent)

        self._collection = collection

        self._ui = load_ui('link_editor.ui', self,
                           directory=os.path.dirname(__file__))

        self._init_widgets()
        self._connect()
        if len(collection) > 1:
            self._ui.right_components.set_data_row(1)
        self._size = None

    def _init_widgets(self):
        self._ui.left_components.setup(self._collection)
        self._ui.right_components.setup(self._collection)
        self._ui.signature_editor.hide()
        print('what is left components', self._ui.left_components)
        for link in self._collection.links:
            self._add_link(link)

    def _connect(self):
        self._ui.add_link.clicked.connect(self._add_new_link)
        self._ui.smart_link.clicked.connect(self._add_smart_link)
        self._ui.remove_link.clicked.connect(self._remove_link)
        self._ui.toggle_editor.clicked.connect(self._toggle_advanced)
        self._ui.signature_editor._ui.addButton.clicked.connect(
            self._add_new_link)

    @property
    def advanced(self):
        return self._ui.signature_editor.isVisible()

    @advanced.setter
    def advanced(self, state):
        """Set whether the widget is in advanced state"""
        self._ui.signature_editor.setVisible(state)
        self._ui.toggle_editor.setText("Basic" if state else "Advanced")

    def _toggle_advanced(self):
        """Show or hide the signature editor widget"""
        self.advanced = not self.advanced

    def _selected_components(self):
        result = []
        id1 = self._ui.left_components.component
        id2 = self._ui.right_components.component
        # print('component', self._ui.left_components, self._ui.left_components.)
        # print()
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

    def _add_link(self, link):
        current = self._ui.current_links
        item = QtWidgets.QListWidgetItem(str(link))
        current.addItem(item)
        item.setHidden(link.hidden)
        current.set_data(item, link)

    def _add_new_link(self):
        if not self.advanced:
            links = self._simple_links()
        else:
            links = self._ui.signature_editor.links()
            self._ui.signature_editor.clear_inputs()

        for link in links:
            self._add_link(link)

    def _add_smart_link(self):
        # load table here
        # TODO: add support for advanced linking mode
        if not self.advanced:
            # TODO: links = linked_conponents_list which is get from the list file
            l = self._ui.left_components # datacollection[0] or sth
            r = self._ui.right_components

            link_file = [['lat', 'latitude'], ['lon', 'longitude'], ['Declination', 'Vopt']]
            # assume we have a ascii table link_data.dat
            for each_list in link_file:
                for i in range(l.count):
                    l_item = l.get_item(i)
                    if l_item.text() in each_list:
                        l_comp = l.get_component(l_item)
                        for j in range(r.count):
                            r_item = r.get_item(j)
                            print('r_item text', r_item.text())
                            print('each list', each_list)
                            if r_item.text() in each_list:
                                r_comp = r.get_component(r_item)
                                assert isinstance(l_comp, core.data.ComponentID), l_comp
                                assert isinstance(r_comp, core.data.ComponentID), r_comp

                                link = core.component_link.ComponentLink([l_comp], r_comp)
                                print('find smart link!', link)
                                self._add_link(link)


            '''for each_list in link_file:
                for left_cid in left_cids:
                    if left_cid.label in each_list:
                        print('what is left_cid.label', left_cid.label)
                        for right_cid in right_cids:
                            if right_cid.label in each_list:
                                print('what is right_cid.label', right_cid.label)
                                # link_pair.append([left_cid.label, right_cid.label])
                                # left_item = QtWidgets.QListWidgetItem(left_cid.label)
                                # right_item = QtWidgets.QListWidgetItem(right_cid.label)

                                print('left_item', left_item)
                                # left_comp = self._ui.left_components.get_component(left_item)
                                # right_comp = self._ui.right_components.get_component(right_item)

                                assert isinstance(left_comp, core.data.ComponentID), left_comp
                                assert isinstance(right_comp, core.data.ComponentID), right_comp

                                link = core.component_link.ComponentLink([left_comp], right_comp)
                                print('find smart link!', link)
                                self._add_link(link)'''


                # component_list = data.components

            # links.append(self._simple_links())  # add user manual linked items too
            self._update_smart_link_list()
        else:
            return
        # assert len(links) != 0
        # for link in links:
        #     self._add_link(link)
        # pass

    def _update_smart_link_list(self):
        #TODO: add update
        pass

    def links(self):
        current = self._ui.current_links
        return current.data.values()

    def _remove_link(self):
        current = self._ui.current_links
        item = current.currentItem()
        row = current.currentRow()
        if item is None:
            return
        current.drop_data(item)
        deleted = current.takeItem(row)
        assert deleted == item  # sanity check

    @classmethod
    def update_links(cls, collection):
        widget = cls(collection)
        isok = widget._ui.exec_()
        if isok:
            links = widget.links()
            collection.set_links(links)


def main():
    import numpy as np
    from glue.utils.qt import get_qapp
    from glue.core import Data, DataCollection

    app = get_qapp()

    x = np.array([1, 2, 3])
    d = Data(label='data', x=x, y=x * 2)
    dc = DataCollection(d)

    LinkEditor.update_links(dc)

if __name__ == "__main__":
    main()
