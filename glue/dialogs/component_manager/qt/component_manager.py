from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict, Counter

from qtpy import QtWidgets, QtGui
from qtpy.QtCore import Qt

from glue.external.echo import SelectionCallbackProperty
from glue.external.echo.qt import connect_combo_selection
from glue.utils.qt import load_ui

__all__ = ['ComponentManagerWidget']


class ComponentManagerWidget(QtWidgets.QDialog):

    data = SelectionCallbackProperty()

    def __init__(self, data_collection=None, initial_data=None, parent=None):

        super(ComponentManagerWidget, self).__init__(parent=parent)

        self.ui = load_ui('component_manager.ui', self,
                          directory=os.path.dirname(__file__))

        self.list = {}
        self.list = self.ui.list_main_components

        self.data_collection = data_collection

        self._components_main = defaultdict(list)
        self._components_other = defaultdict(list)
        self._state = defaultdict(dict)

        for data in data_collection:

            for cid in data.main_components:
                comp_state = {}
                comp_state['cid'] = cid
                comp_state['label'] = cid.label
                self._state[data][cid] = comp_state
                self._components_main[data].append(cid)

            # Keep track of all other components

            self._components_other[data] = []

            for cid in data.components:
                if cid not in self._components_main[data]:
                    self._components_other[data].append(cid)

        # Populate data combo
        ComponentManagerWidget.data.set_choices(self, list(self.data_collection))
        ComponentManagerWidget.data.set_display_func(self, lambda x: x.label)
        connect_combo_selection(self, 'data', self.ui.combosel_data)

        if initial_data is None:
            self.ui.combosel_data.setCurrentIndex(0)
        else:
            self.data = initial_data

        self.ui.combosel_data.currentIndexChanged.connect(self._update_component_lists)
        self._update_component_lists()

        self.ui.button_remove_main.clicked.connect(self._remove_main_component)

        self.ui.list_main_components.itemSelectionChanged.connect(self._update_selection_main)

        self._update_selection_main()

        self.ui.list_main_components.itemChanged.connect(self._update_state)
        self.ui.list_main_components.order_changed.connect(self._update_state)

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

    def _update_selection_main(self):
        enabled = self.list.selected_cid is not None
        self.button_remove_main.setEnabled(enabled)

    def _update_component_lists(self, *args):

        # This gets called when the data is changed and we need to update the
        # components shown in the lists.

        self.list.blockSignals(True)

        self.list.clear()
        for cid in self._components_main[self.data]:
            self.list.add_cid_and_label(cid, [self._state[self.data][cid]['label']])

        self.list.blockSignals(False)

        self._validate()

    def _validate(self):

        # Construct a list of all labels for the current dataset so that
        # we can check which ones are duplicates
        labels = [c.label for c in self._components_other[self.data]]
        labels.extend([c['label'] for c in self._state[self.data].values()])
        if len(labels) == 0:
            return
        label_count = Counter(labels)

        # It's possible that the duplicates are entirely for components not
        # shown in this editor, so we keep track here of whether an invalid
        # component has been found.
        invalid = False

        if label_count.most_common(1)[0][1] > 1:

            # If we are here, there are duplicates somewhere in the list
            # of components.

            brush_red = QtGui.QBrush(Qt.red)
            brush_black = QtGui.QBrush(Qt.black)

            self.list.blockSignals(True)

            for item in self.list:
                label = item.text(0)
                if label_count[label] > 1:
                    item.setForeground(0, brush_red)
                    invalid = True
                else:
                    item.setForeground(0, brush_black)

            self.list.blockSignals(False)

        if invalid:
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

        self._components_main[self.data] = []
        for item in self.list:
            cid = item.data(0, Qt.UserRole)
            self._state[self.data][cid]['label'] = item.text(0)
            self._components_main[self.data].append(cid)

        self._update_component_lists()

    def _remove_main_component(self, *args):
        cid = self.list.selected_cid
        if cid is not None:
            self._components_main[self.data].remove(cid)
            self._state[self.data].pop(cid)
            self._update_component_lists()

    def accept(self):

        for data in self._components_main:

            cids_main = self._components_main[data]
            cids_existing = data.components
            cids_all = data.pixel_component_ids + data.world_component_ids + cids_main + data.derived_components

            # First deal with renaming of components
            for cid_new in cids_main:
                label = self._state[data][cid_new]['label']
                if label != cid_new.label:
                    cid_new.label = label

            # Second deal with the removal of components
            for cid_old in cids_existing:
                if not any(cid_old is cid_new for cid_new in cids_all):
                    data.remove_component(cid_old)

            # Findally, reorder components as needed
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
