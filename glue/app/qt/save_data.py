from __future__ import absolute_import, division, print_function

import os

from qtpy.QtWidgets import QDialog, QListWidgetItem
from qtpy.QtCore import Qt

from glue import config

from glue.external.echo import SelectionCallbackProperty
from glue.external.echo.qt import autoconnect_callbacks_to_qt

from glue.utils.qt import load_ui
from glue.core.state_objects import State
from glue.external.echo import ChoiceSeparator
from glue.core.data_combo_helper import ComponentIDComboHelper, DataCollectionComboHelper
from glue.core.data_exporters.qt.dialog import export_data

__all__ = ['SaveDataDialog']


class SaveDataState(State):

    data = SelectionCallbackProperty()
    subset = SelectionCallbackProperty()
    component = SelectionCallbackProperty()
    exporter = SelectionCallbackProperty()

    def __init__(self, data_collection=None):

        super(SaveDataState, self).__init__()

        self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)
        self.component_helper = ComponentIDComboHelper(self, 'component',
                                                       data_collection=data_collection)

        self.add_callback('data', self._on_data_change)
        self._on_data_change()

        self._sync_data_exporters()

    def _sync_data_exporters(self):

        exporters = list(config.data_exporter)

        def display_func(exporter):
            if exporter.extension == '':
                return "{0} (*)".format(exporter.label)
            else:
                return "{0} ({1})".format(exporter.label, ' '.join('*.' + ext for ext in exporter.extension))

        SaveDataState.exporter.set_choices(self, exporters)
        SaveDataState.exporter.set_display_func(self, display_func)

    def _on_data_change(self, event=None):
        self.component_helper.set_multiple_data([self.data])
        self._sync_subsets()

    def _sync_subsets(self):

        def display_func(subset):
            if subset is None:
                return "All data (no subsets applied)"
            else:
                return subset.label

        subsets = [None] + list(self.data.subsets)

        SaveDataState.subset.set_choices(self, subsets)
        SaveDataState.subset.set_display_func(self, display_func)


class SaveDataDialog(QDialog):

    def __init__(self, data_collection=None, parent=None):

        super(SaveDataDialog, self).__init__(parent=parent)

        self.state = SaveDataState(data_collection=data_collection)

        self.ui = load_ui('save_data.ui', parent=self,
                          directory=os.path.dirname(__file__))
        autoconnect_callbacks_to_qt(self.state, self.ui)

        self.ui.button_cancel.clicked.connect(self.reject)
        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_select_none.clicked.connect(self.select_none)
        self.ui.button_select_all.clicked.connect(self.select_all)

        self.ui.list_component.itemChanged.connect(self._on_check_change)

        self.state.add_callback('component', self._on_data_change)

        self._on_data_change()

    def _on_data_change(self, *event):

        components = getattr(type(self.state), 'component').get_choices(self.state)

        self.ui.list_component.clear()

        for component in components:

            if isinstance(component, ChoiceSeparator):
                item = QListWidgetItem(str(component))
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                item.setForeground(Qt.gray)
            else:
                item = QListWidgetItem(component.label)
                item.setCheckState(Qt.Checked)
            self.ui.list_component.addItem(item)

    def _on_check_change(self, *event):

        any_checked = False

        for idx in range(self.ui.list_component.count()):
            item = self.ui.list_component.item(idx)
            if item.checkState() == Qt.Checked:
                any_checked = True
                break

        self.button_ok.setEnabled(any_checked)

    def select_none(self, *event):
        self._set_all_checked(False)

    def select_all(self, *event):
        self._set_all_checked(True)

    def _set_all_checked(self, check_state):
        for idx in range(self.ui.list_component.count()):
            item = self.ui.list_component.item(idx)
            item.setCheckState(Qt.Checked if check_state else Qt.Unchecked)

    def accept(self):
        components = []
        for idx in range(self.ui.list_component.count()):
            item = self.ui.list_component.item(idx)
            if item.checkState() == Qt.Checked:
                components.append(self.state.data.id[item.text()])
        if self.state.subset is None:
            data = self.state.data
        else:
            data = self.state.subset
        export_data(data, components=components, exporter=self.state.exporter.function)
        super(SaveDataDialog, self).accept()


if __name__ == "__main__":

    from glue.core import DataCollection, Data
    from glue.utils.qt import get_qapp

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
    data2 = Data(a=[1, 2, 3], b=[2, 3, 4], label='data2')
    dc = DataCollection([data1, data2])

    app = get_qapp()

    dialog = SaveDataDialog(data_collection=dc)
    dialog.exec_()
