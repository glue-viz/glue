from __future__ import absolute_import, division, print_function

from ..table_widget import DataTableModel
from mock import MagicMock

from ..table_widget import DataTableModel, TableWidget
from ....core import Data

from ....external.qt.QtCore import Qt
from . import simple_session


class TestDataTableModel(object):

    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        self.model = DataTableModel(self.data)

    def test_column_count(self):
        assert self.model.columnCount() == 2

    def test_column_count_hidden(self):
        self.model.show_hidden = True
        assert self.model.columnCount() == 4

    def test_header_data(self):
        for i, c in enumerate(self.data.visible_components):
            result = self.model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
            assert result == c.label

        for i in range(self.data.size):
            result = self.model.headerData(i, Qt.Vertical, Qt.DisplayRole)
            assert result == str(i)

    def test_row_count(self):
        assert self.model.rowCount() == 4

    def test_data(self):
        for i, c in enumerate(self.data.visible_components):
            for j in range(self.data.size):
                idx = self.model.index(j, i)
                result = self.model.data(idx, Qt.DisplayRole)
                assert float(result) == self.data[c, j]

    def test_data_2d(self):
        self.data = Data(x=[[1, 2], [3, 4]], y=[[2, 3], [4, 5]])
        self.model = DataTableModel(self.data)
        for i, c in enumerate(self.data.visible_components):
            for j in range(self.data.size):
                idx = self.model.index(j, i)
                result = self.model.data(idx, Qt.DisplayRole)
                assert float(result) == self.data[c].ravel()[j]

    def test_background_color_from_subset(self):
        s = self.data.new_subset()
        s.subset_state = self.data.id['x'] > 2
        s.style.color = '#ff00ff'
        for i in range(2):
            for j in range(4):
                print i, j
                idx = self.model.index(j, i)
                color = self.model.data(idx, Qt.BackgroundColorRole)
                if j < 2:
                    assert color is None
                else:
                    assert color is not None
                    assert color.red() == 255
                    assert color.green() == 0
                    assert color.blue() == 255

    def test_user_role(self):
        idx = self.model.index(0, 0)
        expected = self.data[self.model.columns[0]][0]
        assert self.model.data(idx, Qt.UserRole) == expected

    def test_table_synced_on_subset_update(self):
        s = simple_session()
        s.data_collection.append(self.data)
        sg = s.data_collection.new_subset_group()

        w = TableWidget(s)
        w.sync = MagicMock()
        w.register_to_hub(s.hub)

        sg.style.color = 'blue'
        assert w.sync.call_count == 1
