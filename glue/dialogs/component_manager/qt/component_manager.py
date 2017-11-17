from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict, Counter

from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtCore import Qt

from glue.core import ComponentID
from glue.core.parse import ParsedComponentLink, ParsedCommand
from glue.utils.qt import load_ui

from glue.dialogs.component_manager.qt.equation_editor import EquationEditorDialog

__all__ = ['ComponentManagerWidget']


class ComponentTreeWidget(QtWidgets.QTreeWidget):

    order_changed = QtCore.Signal()

    def dropEvent(self, event):
        items = self.selectedItems()
        selected = items[0] if len(items) == 1 else None
        super(ComponentTreeWidget, self).dropEvent(event)
        self.selection = self.selectionModel()
        self.selection.select(QtCore.QItemSelection(self.indexFromItem(selected, 0),
                                                    self.indexFromItem(selected, self.columnCount() - 1)),
                              QtCore.QItemSelectionModel.ClearAndSelect)
        self.order_changed.emit()

    def mousePressEvent(self, event):
        self.clearSelection()
        super(ComponentTreeWidget, self).mousePressEvent(event)


class ComponentManagerWidget(QtWidgets.QDialog):

    def __init__(self, data_collection=None, parent=None):

        super(ComponentManagerWidget, self).__init__(parent=parent)

        self.ui = load_ui('component_manager.ui', self,
                          directory=os.path.dirname(__file__))

        self.data_collection = data_collection

        self._components = defaultdict(lambda: defaultdict(list))
        self._state = defaultdict(dict)

        for data in data_collection:

            for cid in data.primary_components:
                comp_state = {}
                comp_state['cid'] = cid
                comp_state['label'] = cid.label
                self._state[data][cid] = comp_state
                self._components[data]['main'].append(cid)

            self._components[data]['derived'] = []

            for cid in data.derived_components:
                comp = data.get_component(cid)
                if isinstance(comp.link, ParsedComponentLink):
                    comp_state = {}
                    comp_state['cid'] = cid
                    comp_state['label'] = cid.label
                    comp_state['equation'] = comp.link._parsed._cmd
                    self._state[data][cid] = comp_state
                    self._components[data]['derived'].append(cid)

        # Populate data combo
        for data in self.data_collection:
            self.ui.combosel_data.addItem(data.label, userData=data)

        self.ui.combosel_data.setCurrentIndex(0)
        self.ui.combosel_data.currentIndexChanged.connect(self._update_component_lists)
        self._update_component_lists()

        self.ui.button_remove_main.clicked.connect(self._remove_main_component)

        self.ui.button_add_derived.clicked.connect(self._add_derived_component)
        self.ui.button_edit_derived.clicked.connect(self._edit_derived_component)
        self.ui.button_remove_derived.clicked.connect(self._remove_derived_component)

        self.ui.list_main_components.itemSelectionChanged.connect(self._update_selection_main)
        self.ui.list_derived_components.itemSelectionChanged.connect(self._update_selection_derived)

        self._update_selection_main()
        self._update_selection_derived()

        self.ui.list_main_components.itemChanged.connect(self._update_state)
        self.ui.list_derived_components.itemChanged.connect(self._update_state)
        self.ui.list_main_components.order_changed.connect(self._update_state)
        self.ui.list_derived_components.order_changed.connect(self._update_state)

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

    def _update_selection_main(self):
        enabled = self.selected_main_component is not None
        self.button_remove_main.setEnabled(enabled)

    def _update_selection_derived(self):
        enabled = self.selected_derived_component is not None
        self.button_edit_derived.setEnabled(enabled)
        self.button_remove_derived.setEnabled(enabled)

    @property
    def data(self):
        return self.ui.combosel_data.currentData()

    @property
    def selected_main_component(self):
        items = self.ui.list_main_components.selectedItems()
        if len(items) == 1:
            return items[0].data(0, Qt.UserRole)
        else:
            return None

    @property
    def selected_derived_component(self):
        items = self.ui.list_derived_components.selectedItems()
        if len(items) == 1:
            return items[0].data(0, Qt.UserRole)
        else:
            return None

    def _update_component_lists(self, *args):

        # This gets called when the data is changed and we need to update the
        # components shown in the lists.

        self.ui.list_main_components.blockSignals(True)

        self.ui.list_main_components.clear()
        self.ui.list_derived_components.clear()

        root = self.ui.list_main_components.invisibleRootItem()

        for cid in self._components[self.data]['main']:
            item = QtWidgets.QTreeWidgetItem(root, [self._state[self.data][cid]['label']])
            item.setData(0, Qt.UserRole, cid)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setFlags(item.flags() ^ Qt.ItemIsDropEnabled)

        self.ui.list_main_components.blockSignals(False)

        self.ui.list_derived_components.blockSignals(True)

        root = self.ui.list_derived_components.invisibleRootItem()

        for cid in self._components[self.data]['derived']:
            item = QtWidgets.QTreeWidgetItem(root, [self._state[self.data][cid]['label']])
            item.setData(0, Qt.UserRole, cid)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setFlags(item.flags() ^ Qt.ItemIsDropEnabled)

        self.ui.list_derived_components.blockSignals(False)

        self._validate()

    def _validate(self):

        # Figure out a list of all the labels so that we can check which ones
        # are duplicates.
        print(list(self._state[self.data].values()))
        labels = [c['label'] for c in self._state[self.data].values()]
        label_count = Counter(labels)

        brush_red = QtGui.QBrush(Qt.red)
        brush_black = QtGui.QBrush(Qt.black)

        if label_count.most_common(1)[0][1] > 1:
            for component_list in (self.ui.list_main_components,
                                   self.ui.list_derived_components):
                component_list.blockSignals(True)
                root = component_list.invisibleRootItem()
                for idx in range(root.childCount()):
                    item = root.child(idx)
                    if label_count[item.text(0)] > 1:
                        item.setForeground(0, brush_red)
                    else:
                        item.setForeground(0, brush_black)
                component_list.blockSignals(False)

            self.ui.label_status.setStyleSheet('color: red')
            self.ui.label_status.setText('Error: some components have duplicate names')
            self.ui.button_ok.setEnabled(False)
            self.ui.combosel_data.setEnabled(False)
            return

        self.ui.label_status.setStyleSheet('')
        self.ui.label_status.setText('')
        self.ui.button_ok.setEnabled(True)
        self.ui.combosel_data.setEnabled(True)

    def _update_state(self, *args):

        self._components[self.data]['main'] = []
        root = self.ui.list_main_components.invisibleRootItem()
        for idx in range(root.childCount()):
            item = root.child(idx)
            cid = item.data(0, Qt.UserRole)
            self._state[self.data][cid]['label'] = item.text(0)
            self._components[self.data]['main'].append(cid)

        self._components[self.data]['derived'] = []
        root = self.ui.list_derived_components.invisibleRootItem()
        for idx in range(root.childCount()):
            item = root.child(idx)
            cid = item.data(0, Qt.UserRole)
            self._state[self.data][cid]['label'] = item.text(0)
            self._components[self.data]['derived'].append(cid)

        self._update_component_lists()

    def _remove_main_component(self, *args):
        cid = self.selected_main_component
        if cid is None:
            return
        self._components[self.data]['main'].remove(cid)
        self._state[self.data].pop(cid)
        self._update_component_lists()

    def _remove_derived_component(self, *args):
        cid = self.selected_derived_component
        if cid is None:
            return
        self._components[self.data]['derived'].remove(cid)
        self._state[self.data].pop(cid)
        self._update_component_lists()

    def _add_derived_component(self, *args):

        comp_state = {}
        comp_state['cid'] = ComponentID('')
        comp_state['label'] = 'New component'
        comp_state['equation'] = ''

        self._components[self.data]['derived'].append(comp_state['cid'])
        self._state[self.data][comp_state['cid']] = comp_state

        self._update_component_lists()

    def _edit_derived_component(self, *args):

        cid = self.selected_derived_component

        dialog = EquationEditorDialog(self.data, self._state[self.data][cid]['equation'], parent=self)
        dialog.setWindowFlags(self.windowFlags() | Qt.Window)
        dialog.setFocus()
        dialog.raise_()
        dialog.exec_()

        if dialog.final_expression is None:
            return

        self._state[self.data][cid]['equation'] = dialog.final_expression

    def accept(self):

        for data in self._components:

            cids_main = self._components[data]['main']
            cids_derived = self._components[data]['derived']

            cids_all = cids_main + cids_derived

            cids_existing = data.components

            for cid_old in cids_existing:
                if not any(cid_old is cid_new for cid_new in cids_all):
                    data.remove_component(cid_old)

            # TODO: make it so labels in expression take into account renaming
            components = dict((cid.label, cid) for cid in data.components)

            for cid_new in cids_derived:
                if not any(cid_new is cid_old for cid_old in cids_existing):
                    pc = ParsedCommand(self._state[data][cid_new]['equation'], components)
                    link = ParsedComponentLink(cid_new, pc)
                    data.add_component_link(link)

            data.reorder_components(cids_all)

        super(ComponentManagerWidget, self).accept()


def main():

    import numpy as np

    from glue.core.data import Data
    from glue.core.data_collection import DataCollection

    x = np.random.random((5, 5))
    y = x * 3
    dc = DataCollection()
    dc.append(Data(label='test1', x=x, y=y))
    dc.append(Data(label='test2', a=x, b=y))

    widget = ComponentManagerWidget(dc)
    widget.exec_()

    for data in dc:
        print('-' * 72)
        print(data)


if __name__ == "__main__":
    from glue.utils.qt import get_qapp
    app = get_qapp()
    main()
