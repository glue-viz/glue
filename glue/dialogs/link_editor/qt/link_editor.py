from __future__ import absolute_import, division, print_function

import os

from qtpy import QtWidgets

from glue.config import link_function, link_helper
from glue.utils.decorators import avoid_circular
from glue.utils.qt import load_ui
from glue.external.echo.qt import autoconnect_callbacks_to_qt
from glue.external.echo.qt.connect import UserDataWrapper, connect_combo_selection
from glue.dialogs.link_editor.state import LinkEditorState

__all__ = ['LinkEditor', 'main']


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

        self._ui = load_ui('link_editor_widget.ui', self,
                           directory=os.path.dirname(__file__))
        autoconnect_callbacks_to_qt(self.state, self._ui)

        self._ui.graph_widget.set_data_collection(data_collection, new_links=self.state.links)
        self._ui.graph_widget.selection_changed.connect(self._on_data_change_graph)

        self._menu = LinkMenu(parent=self._ui.button_add_link)
        self._menu.triggered.connect(self._add_link)
        self._ui.button_add_link.setMenu(self._menu)

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
        self._ui.button_remove_link.setEnabled(enabled)

    def _on_current_link_change(self, *args):

        # We update the link details panel on the right

        link_details = self._ui.link_details

        link_io_widget = QtWidgets.QWidget()
        link_io = QtWidgets.QGridLayout()
        link_io_widget.setLayout(link_io)

        link_io.setSizeConstraint(link_io.SetFixedSize)
        link_io.setHorizontalSpacing(10)
        link_io.setVerticalSpacing(5)
        link_io.setContentsMargins(0, 0, 0, 0)

        item = self._ui.link_io.itemAt(0)
        if item is not None and item.widget() is not None:
            widget = item.widget()
            widget.setParent(None)
            # NOTE: we need to also hide the widget otherwise it will still
            # appear but floating in front of or behind the dialog.
            widget.hide()

        for row in range(link_io.rowCount()):
            link_io.setRowStretch(row, 0.5)

        link_io.setColumnStretch(1, 10)

        link = self.state.current_link

        if link is None:
            link_details.setText('')
            return

        link_details.setText(link.description)

        index = 0

        if link.data1 is self.state.data1:
            data1_names = link.names1
        else:
            data1_names = link.names2

        if len(data1_names) > 0:

            link_io.addWidget(QtWidgets.QLabel('<b>Dataset 1 attributes</b>'), 0, 0, 1, 2)

            for input_name in data1_names:
                index += 1
                combo = QtWidgets.QComboBox(parent=self._ui)
                combo.setMinimumContentsLength(10)
                combo.setSizeAdjustPolicy(combo.AdjustToMinimumContentsLength)
                link_io.addWidget(QtWidgets.QLabel(input_name), index, 0)
                link_io.addWidget(combo, index, 1)
                connect_combo_selection(link, input_name, combo)

        if link.data1 is self.state.data2:
            data2_names = link.names1
        else:
            data2_names = link.names2

        if len(data2_names) > 0:

            index += 1
            link_io.addItem(QtWidgets.QSpacerItem(5, 20,
                                                  QtWidgets.QSizePolicy.Fixed,
                                                  QtWidgets.QSizePolicy.Fixed), index, 0)

            index += 1
            link_io.addWidget(QtWidgets.QLabel('<b>Dataset 2 attributes</b>'), index, 0, 1, 2)

            for output_name in data2_names:
                index += 1
                combo = QtWidgets.QComboBox(parent=self._ui)
                combo.setMinimumContentsLength(10)
                combo.setSizeAdjustPolicy(combo.AdjustToMinimumContentsLength)
                link_io.addWidget(QtWidgets.QLabel(output_name), index, 0)
                link_io.addWidget(combo, index, 1)
                connect_combo_selection(link, output_name, combo)

        index += 1
        link_io.addWidget(QtWidgets.QWidget(), index, 0)
        link_io.setRowStretch(index, 10)

        self._ui.link_io.addWidget(link_io_widget)

        self._ui.graph_widget.set_links(self.state.links)


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
