from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.config import link_function, link_helper
from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.external.echo.qt.connect import UserDataWrapper
from glue.dialogs.link_editor.state import LinkEditorState

__all__ = ['LinkEditor']


def get_function_name(info):
    item = info[0]
    if hasattr(item, 'display') and item.display is not None:
        return item.display
    else:
        return item.__name__


# TODO: make links shallow-copiable so that we avoid changing the real ones in-place
# TODO: make data combos not allow same data to be selected twice
# TODO: make a helper that can show the link editor panel on the right for
# average links. But also need a way to provide a custom one, for example
# for WCSLink.

# Can we change type of an existing link? Or should 'add link' button have a
# drop down for link type? Does this all make identity links harder?

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


class WidgetContainer(object):
    pass


class LinkEditor(QtWidgets.QDialog):

    def __init__(self, data_collection, parent=None):

        super(LinkEditor, self).__init__(parent=parent)

        self._data_collection = data_collection

        # TODO: This is the point where we should actually take copies of the
        # links because we are going to be editing them and the user may cancel
        self._links = list(data_collection.external_links)

        self.state = LinkEditorState(data_collection, self._links)

        self._ui = load_ui('link_editor.ui', self,
                           directory=os.path.dirname(__file__))
        autoconnect_callbacks_to_qt(self.state, self._ui)

        self._ui.graph_widget.set_data_collection(data_collection)
        self._ui.graph_widget.selection_changed.connect(self._on_data_change_graph)

        self._menu = LinkMenu(parent=self._ui.button_add_link)
        self._menu.triggered.connect(self._add_link)
        self._ui.button_add_link.setMenu(self._menu)

        self.state.add_callback('links', self._on_links_change)

    def _add_link(self, action):
        self.state.add_link(action.data().data)

    @avoid_circular
    def _on_data_change_graph(self):
        self.state.data1 = getattr(self._ui.graph_widget.selected_node1, 'data', None)
        self.state.data2 = getattr(self._ui.graph_widget.selected_node2, 'data', None)

    def _on_links_change(self, *args):
        # We update the link details panel on the right

        print('_on_links_change')

        link = self.state.links

        print(link)

        link_io = self._ui.link_io

        for i in reversed(range(link_io.count())):
            item = link_io.itemAt(i)
            if item is not None and item.widget() is not None:
                item.widget().setParent(None)

        for row in range(link_io.rowCount()):
            link_io.setRowStretch(row, 0)

        if link is None:
            return

        container = WidgetContainer()

        # FIXME: some combo widgets appear outside of window too

        link_io.addWidget(QtWidgets.QLabel('<b>Inputs</b>'), 0, 0, 1, 2)

        for index, input_name in enumerate(link._input_names):
            print("adding combo", input_name)
            combo = QtWidgets.QComboBox(parent=self)
            setattr(container, 'combosel_' + input_name, combo)
            link_io.addWidget(QtWidgets.QLabel(input_name), index + 1, 0)
            link_io.addWidget(combo, index + 1, 1)

        link_io.addItem(QtWidgets.QSpacerItem(5, 20,
                                              QtWidgets.QSizePolicy.Fixed,
                                              QtWidgets.QSizePolicy.Fixed),
                        index + 2, 0)

        link_io.addWidget(QtWidgets.QLabel('<b>Output</b>'), index + 3, 0, 1, 2)

        combo = QtWidgets.QComboBox(parent=self)
        setattr(container, 'combosel_' + link._output_name, combo)
        link_io.addWidget(QtWidgets.QLabel(link._output_name), index + 4, 0)
        link_io.addWidget(combo, index + 4, 1)

        link_io.addWidget(QtWidgets.QWidget(), index + 5, 0)

        link_io.setRowStretch(index + 5, 10)

        # TODO: check that we don't have residual references
        autoconnect_callbacks_to_qt(link, container)

    @classmethod
    def update_links(cls, collection):
        widget = cls(collection)
        isok = widget._ui.exec_()
        if isok:
            collection.set_links(widget._links)


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
    from glue.main import load_plugins
    load_plugins()
    main()
