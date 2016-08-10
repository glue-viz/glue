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
from glue.config import auto_linking

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
        """ Return current selected component objects from both left and right list widget.

        :return: a list which contains two `~glue.core.data.ComponentID` objects.
        """
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
            self._add_link(link)  # the link here is comp0 <-> comp1
            comp = self._selected_components()
            # Users' manual link component will be added to the link_array
            flag = False
            for i in range(auto_linking.count):
                if comp[0].label in auto_linking.members[i] and comp[1].label not in auto_linking.members[i]:
                    auto_linking.add_to(comp[1].label, i)
                    flag = True
                    break
                if comp[1].label in auto_linking.members[i] and comp[0].label not in auto_linking.members[i]:
                    auto_linking.add_to(comp[0].label, i)
                    flag = True
                    break
                if comp[0].label in auto_linking.members[i] and comp[1].label in auto_linking.members[i]:
                    flag = True
                    break   # pari already exists

            if not flag:
            # TODO: should not add here, should check all auto_linking members
                from sets import Set
                auto_linking.add(Set([comp[0].label, comp[1].label]))

    def _add_smart_link(self):
        if not self.advanced:
            l = self._ui.left_components
            r = self._ui.right_components

        # TODO: if same dataset then skip

            link_array = auto_linking.members
            for each_list in link_array:
                for i in range(l.count):
                    l_item = l.get_item(i)
                    if l_item.text() in each_list:
                        l_comp = l.get_component(l_item)
                        for j in range(r.count):
                            r_item = r.get_item(j)
                            if r_item.text() in each_list:
                                r_comp = r.get_component(r_item)
                                assert isinstance(l_comp, core.data.ComponentID), l_comp
                                assert isinstance(r_comp, core.data.ComponentID), r_comp

                                link = core.component_link.ComponentLink([l_comp], r_comp)
                                self._add_link(link)

        else:
            # TODO: advanced link mode?
            return

    def links(self):
        current = self._ui.current_links
        return current.data.values()

    # TODO: unglue from result panel can't be traced by the code now
    def _remove_link(self):
        current = self._ui.current_links
        item = current.currentItem()
        row = current.currentRow()
        if item is None:
            return
        current.drop_data(item)
        deleted = current.takeItem(row)
        assert deleted == item  # sanity check


        '''
        # TODO: find way to get link objects in item
        comp = self._selected_components()
        # Users' manual remove component will also get removed to the link_array
        for i in range(auto_linking.count):
            print('auto_linking.members[i]', auto_linking.members[i], 'label', comp[0].label)
            if comp[0].label in auto_linking.members[i]:
                print('auto_linking.members[i]', auto_linking.members[i], 'label', comp[0].label)
                auto_linking.remove_from(comp[0].label, i)
            if comp[1].label in auto_linking.members[i]:
                auto_linking.remove_from(comp[1].label, i)'''


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
