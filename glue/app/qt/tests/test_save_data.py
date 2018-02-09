from collections import namedtuple

from mock import MagicMock, patch

from qtpy.QtCore import Qt

from glue.core import DataCollection, Data
from glue.utils.qt import get_qapp
from glue.app.qt.save_data import SaveDataDialog


def components(list_widget):
    enabled = []
    disabled = []
    for idx in range(list_widget.count()):
        item = list_widget.item(idx)
        if item.checkState() == Qt.Checked:
            enabled.append(item.text())
        else:
            disabled.append(item.text())
    return disabled, enabled


class TestSaveDataDialog:

    def setup_method(self, method):

        self.data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
        self.data2 = Data(a=[1, 2, 3], b=[2, 3, 4], label='data2')
        self.dc = DataCollection([self.data1, self.data2])

        self.x = self.data1.id['x']
        self.y = self.data1.id['y']
        self.a = self.data2.id['a']
        self.b = self.data2.id['b']

        self.app = get_qapp()

        self.dialog = SaveDataDialog(data_collection=self.dc)

    def test_defaults(self):
        disabled, enabled = components(self.dialog.ui.list_component)
        assert enabled == ['x', 'y']
        assert disabled == []

    def test_change_data(self):
        self.dialog.ui.combosel_data.setCurrentIndex(1)
        disabled, enabled = components(self.dialog.ui.list_component)
        assert enabled == ['a', 'b']
        assert disabled == []

    def test_select_buttons(self):
        self.dialog.button_select_none.click()
        disabled, enabled = components(self.dialog.ui.list_component)
        assert enabled == []
        assert disabled == ['x', 'y']
        self.dialog.button_select_all.click()
        disabled, enabled = components(self.dialog.ui.list_component)
        assert enabled == ['x', 'y']
        assert disabled == []



    def test_accept(self):
        func = self._accept()
        func.assert_called_once_with('test_file.fits', self.data1, components=[self.x, self.y])

    def test_change_accept(self):
        self.dialog.ui.combosel_data.setCurrentIndex(1)
        func = self._accept()
        func.assert_called_once_with('test_file.fits', self.data2, components=[self.a, self.b])

    def test_deselect_accept(self):
        self.dialog.ui.list_component.item(1).setCheckState(Qt.Unchecked)
        func = self._accept()
        func.assert_called_once_with('test_file.fits', self.data1, components=[self.x])

    def test_deselect_all(self):
        self.dialog.select_none()
        assert not self.dialog.button_ok.isEnabled()
        self.dialog.select_all()
        assert self.dialog.button_ok.isEnabled()

    def _accept(self):

        mock = MagicMock()

        test_exporter_cls = namedtuple('exporter', 'function label extension')
        test_exporter = test_exporter_cls(function=mock, label='Test', extension='')

        with patch('qtpy.compat.getsavefilename') as dialog:
            with patch('glue.config.data_exporter') as data_exporter:
                def test_iter(x):
                    yield test_exporter
                data_exporter.__iter__ = test_iter
                dialog.return_value = 'test_file.fits', 'Test (*)'
                self.dialog.accept()
        return test_exporter.function



# import os
#
# from qtpy.QtWidgets import QDialog, QListWidgetItem
# from qtpy.QtCore import Qt
#
# from glue.external.echo import SelectionCallbackProperty
# from glue.external.echo.qt import autoconnect_callbacks_to_qt
#
# from glue.utils.qt import load_ui
# from glue.core.state_objects import State
# from glue.core.data_combo_helper import ComponentIDComboHelper, DataCollectionComboHelper
# from glue.core.data_exporters.qt.dialog import export_data
#
# __all__ = ['SaveDataDialog']
#
#
# class SaveDataState(State):
#
#     data = SelectionCallbackProperty()
#     component = SelectionCallbackProperty()
#
#     def __init__(self, data_collection=None):
#
#         super(SaveDataState, self).__init__()
#
#         self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)
#         self.component_helper = ComponentIDComboHelper(self, 'component',
#                                                        data_collection=data_collection)
#
#         self.add_callback('data', self._on_data_change)
#         self._on_data_change()
#
#     def _on_data_change(self, event=None):
#         self.component_helper.set_multiple_data([self.data])
#
#
# class SaveDataDialog(QDialog):
#
#     def __init__(self, data_collection=None, parent=None):
#
#         super(SaveDataDialog, self).__init__(parent=parent)
#
#         self.state = SaveDataState(data_collection=data_collection)
#
#         self.ui = load_ui('save_data.ui', parent=self,
#                           directory=os.path.dirname(__file__))
#         autoconnect_callbacks_to_qt(self.state, self)
#
#         self.ui.button_cancel.clicked.connect(self.reject)
#         self.ui.button_ok.clicked.connect(self.accept)
#         self.ui.button_select_none.clicked.connect(self.select_none)
#         self.ui.button_select_all.clicked.connect(self.select_all)
#
#         self.state.add_callback('component', self._on_data_change)
#
#         self._on_data_change()
#
#     def _on_data_change(self, *event):
#
#         components = getattr(type(self.state), 'component').get_choices(self.state)
#
#         self.ui.list_component.clear()
#
#         for component in components:
#
#             item = QListWidgetItem(component.label)
#             item.setCheckState(Qt.Checked)
#             self.ui.list_component.addItem(item)
#
#     def select_none(self, *event):
#         self._set_all_checked(False)
#
#     def select_all(self, *event):
#         self._set_all_checked(True)
#
#     def _set_all_checked(self, check_state):
#         for idx in range(self.ui.list_component.count()):
#             item = self.ui.list_component.item(idx)
#             item.setCheckState(Qt.Checked if check_state else Qt.Unchecked)
#
#     def accept(self):
#         components = []
#         for idx in range(self.ui.list_component.count()):
#             item = self.ui.list_component.item(idx)
#             if item.checkState() == Qt.Checked:
#                 components.append(self.state.data.id[item.text()])
#         export_data(self.state.data, components=components)
#         super(SaveDataDialog, self).accept()
#
#
# if __name__ == "__main__":
#
#     from glue.core import DataCollection, Data
#     from glue.utils.qt import get_qapp
#
#     data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
#     data2 = Data(a=[1, 2, 3], b=[2, 3, 4], label='data2')
#     dc = DataCollection([data1, data2])
#
#     app = get_qapp()
#
#     dialog = SaveDataDialog(data_collection=dc)
#     dialog.exec_()
