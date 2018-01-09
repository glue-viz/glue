from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict, Counter

from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtCore import Qt

from glue.core import ComponentID
from glue.core.parse import ParsedComponentLink, ParsedCommand
from glue.utils.qt import load_ui
from glue.core.message import NumericalDataChangedMessage

from glue.dialogs.component_manager.qt.equation_editor import EquationEditorDialog

__all__ = ['ComponentManagerWidget']


class ComponentTreeWidget(QtWidgets.QTreeWidget):

    order_changed = QtCore.Signal()

    def select_cid(self, cid):
        for item in self:
            if item.data(0, Qt.UserRole) is cid:
                self.select_item(item)
                return
        raise ValueError("Could not find find cid {0} in list".format(cid))

    def select_item(self, item):
        self.selection = self.selectionModel()
        self.selection.select(QtCore.QItemSelection(self.indexFromItem(item, 0),
                                                    self.indexFromItem(item, self.columnCount() - 1)),
                              QtCore.QItemSelectionModel.ClearAndSelect)

    @property
    def selected_item(self):
        items = self.selectedItems()
        return items[0] if len(items) == 1 else None

    @property
    def selected_cid(self):
        selected = self.selected_item
        return None if selected is None else selected.data(0, Qt.UserRole)

    def add_cid_and_label(self, cid, label):
        item = QtWidgets.QTreeWidgetItem(self.invisibleRootItem(), [label])
        item.setData(0, Qt.UserRole, cid)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setFlags(item.flags() ^ Qt.ItemIsDropEnabled)

    def __iter__(self):
        root = self.invisibleRootItem()
        for idx in range(root.childCount()):
            yield root.child(idx)

    def __len__(self):
        return self.invisibleRootItem().childCount()

    def dropEvent(self, event):
        selected = self.selected_item
        super(ComponentTreeWidget, self).dropEvent(event)
        self.select_item(selected)
        self.order_changed.emit()

    def mousePressEvent(self, event):
        self.clearSelection()
        super(ComponentTreeWidget, self).mousePressEvent(event)


class ComponentManagerWidget(QtWidgets.QDialog):

    def __init__(self, data_collection=None, parent=None):

        super(ComponentManagerWidget, self).__init__(parent=parent)

        self.ui = load_ui('component_manager.ui', self,
                          directory=os.path.dirname(__file__))

        self.list = {}
        self.list['main'] = self.ui.list_main_components
        self.list['derived'] = self.ui.list_derived_components

        self.data_collection = data_collection

        self._components = defaultdict(lambda: defaultdict(list))
        self._state = defaultdict(dict)

        for data in data_collection:

            for cid in data.primary_components:
                if not cid.hidden:
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
                    comp_state['equation'] = comp.link._parsed
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
        enabled = self.list['main'].selected_cid is not None
        self.button_remove_main.setEnabled(enabled)

    def _update_selection_derived(self):
        enabled = self.list['derived'].selected_cid is not None
        self.button_edit_derived.setEnabled(enabled)
        self.button_remove_derived.setEnabled(enabled)

    @property
    def data(self):
        try:
            return self.ui.combosel_data.currentData()
        except AttributeError:  # PyQt4
            return self.ui.combosel_data.itemData(self.ui.combosel_data.currentIndex())

    def _update_component_lists(self, *args):

        # This gets called when the data is changed and we need to update the
        # components shown in the lists.

        for component_list in ('main', 'derived'):

            self.list[component_list].blockSignals(True)

            self.list[component_list].clear()
            for cid in self._components[self.data][component_list]:
                self.list[component_list].add_cid_and_label(cid, self._state[self.data][cid]['label'])

            self.list[component_list].blockSignals(False)

        self._validate()

    def _validate(self):

        # Construct a list of all labels for the current dataset so that
        # we can check which ones are duplicates
        labels = [c['label'] for c in self._state[self.data].values()]
        if len(labels) == 0:
            return
        label_count = Counter(labels)

        if label_count.most_common(1)[0][1] > 1:

            # If we are here, there are duplicates somewhere in the list
            # of components.

            brush_red = QtGui.QBrush(Qt.red)
            brush_black = QtGui.QBrush(Qt.black)

            for component_list in ('main', 'derived'):

                self.list[component_list].blockSignals(True)

                for item in self.list[component_list]:
                    label = item.text(0)
                    if label_count[label] > 1:
                        item.setForeground(0, brush_red)
                    else:
                        item.setForeground(0, brush_black)

                self.list[component_list].blockSignals(False)

            self.ui.label_status.setStyleSheet('color: red')
            self.ui.label_status.setText('Error: some components have duplicate names')
            self.ui.button_ok.setEnabled(False)
            self.ui.combosel_data.setEnabled(False)

        else:

            self.ui.label_status.setStyleSheet('')
            self.ui.label_status.setText('')
            self.ui.button_ok.setEnabled(True)
            self.ui.combosel_data.setEnabled(True)

    def _update_state(self, *args):

        for component_list in ('main', 'derived'):

            self._components[self.data][component_list] = []
            for item in self.list[component_list]:
                cid = item.data(0, Qt.UserRole)
                self._state[self.data][cid]['label'] = item.text(0)
                self._components[self.data][component_list].append(cid)

        self._update_component_lists()

    def _remove_main_component(self, *args):
        cid = self.list['main'].selected_cid
        if cid is not None:
            self._components[self.data]['main'].remove(cid)
            self._state[self.data].pop(cid)
            self._update_component_lists()

    def _remove_derived_component(self, *args):
        cid = self.list['derived'].selected_cid
        if cid is not None:
            self._components[self.data]['derived'].remove(cid)
            self._state[self.data].pop(cid)
            self._update_component_lists()

    def _add_derived_component(self, *args):

        comp_state = {}
        comp_state['cid'] = ComponentID('')
        comp_state['label'] = ''
        comp_state['equation'] = None

        self._components[self.data]['derived'].append(comp_state['cid'])
        self._state[self.data][comp_state['cid']] = comp_state

        self._update_component_lists()

        self.list['derived'].select_cid(comp_state['cid'])

        result = self._edit_derived_component()

        if not result:  # user cancelled
            self._components[self.data]['derived'].remove(comp_state['cid'])
            self._state[self.data].pop(comp_state['cid'])
            self._update_component_lists()

    def _edit_derived_component(self, event=None):

        mapping = {}
        references = {}
        for cid in self._components[self.data]['main']:
            label = self._state[self.data][cid]['label']
            mapping[cid] = label
            references[label] = cid

        cid = self.list['derived'].selected_cid
        item = self.list['derived'].selected_item

        if item is None:
            return False

        label = self._state[self.data][cid]['label']

        if self._state[self.data][cid]['equation'] is None:
            equation = None
        else:
            equation = self._state[self.data][cid]['equation'].render(mapping)

        dialog = EquationEditorDialog(label=label, equation=equation, references=references, parent=self)
        dialog.setWindowFlags(self.windowFlags() | Qt.Window)
        dialog.setFocus()
        dialog.raise_()
        dialog.exec_()

        if dialog.final_expression is None:
            return False

        name, equation = dialog.get_final_label_and_parsed_command()
        self._state[self.data][cid]['label'] = name
        self._state[self.data][cid]['equation'] = equation
        item.setText(0, name)

        return True

    def accept(self):

        for data in self._components:

            cids_main = self._components[data]['main']
            cids_derived = self._components[data]['derived']

            # First deal with renaming of components
            for cid_new in cids_main + cids_derived:
                label = self._state[data][cid_new]['label']
                if label != cid_new.label:
                    cid_new.label = label

            cids_all = data.pixel_component_ids + data.world_component_ids + cids_main + cids_derived

            cids_existing = data.components

            for cid_old in cids_existing:
                if not any(cid_old is cid_new for cid_new in cids_all):
                    data.remove_component(cid_old)

            components = dict((cid.uuid, cid) for cid in data.components)

            for cid_new in cids_derived:
                if any(cid_new is cid_old for cid_old in cids_existing):
                    comp = data.get_component(cid_new)
                    if comp.link._parsed._cmd != self._state[data][cid_new]['equation']._cmd:
                        comp.link._parsed._cmd = self._state[data][cid_new]['equation']._cmd
                        comp.link._parsed._references = components
                        if data.hub:
                            msg = NumericalDataChangedMessage(data)
                            data.hub.broadcast(msg)
                else:
                    pc = ParsedCommand(self._state[data][cid_new]['equation']._cmd, components)
                    link = ParsedComponentLink(cid_new, pc)
                    data.add_component_link(link)

            data.reorder_components(cids_all)

        super(ComponentManagerWidget, self).accept()


if __name__ == "__main__":  # pragma: nocover

    from glue.utils.qt import get_qapp
    app = get_qapp()

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
