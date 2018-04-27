from __future__ import absolute_import, division, print_function

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
        if item.flags() & Qt.ItemIsSelectable:
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
        self.dc.new_subset_group(label='my subset', subset_state=self.data1.id['x'] > 1.5)

        self.x = self.data1.id['x']
        self.y = self.data1.id['y']
        self.a = self.data2.id['a']
        self.b = self.data2.id['b']

        self.app = get_qapp()

        self.dialog = SaveDataDialog(data_collection=self.dc)

    def teardown_method(self, method):
        self.app = None

    def test_defaults(self):
        disabled, enabled = components(self.dialog.ui.list_component)
        assert enabled == ['x', 'y']
        assert disabled == []

    def test_defaults_derived(self):
        self.data1['z'] = self.data1.id['x'] + 1
        disabled, enabled = components(self.dialog.ui.list_component)
        assert enabled == ['x', 'y', 'z']
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

    def test_change_subset_accept(self):
        self.dialog.ui.combosel_subset.setCurrentIndex(1)
        func = self._accept()
        func.assert_called_once_with('test_file.fits', self.data1.subsets[0], components=[self.x, self.y])

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
                dialog.return_value = 'test_file.fits', None
                self.dialog.state._sync_data_exporters()
                self.dialog.accept()
        return test_exporter.function
