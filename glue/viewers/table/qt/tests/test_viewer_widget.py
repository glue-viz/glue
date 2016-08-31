from __future__ import absolute_import, division, print_function

import pytest

from qtpy.QtCore import Qt
from glue.core import Data, DataCollection, Session
from glue.utils.qt import qt4_to_mpl_color

from ..viewer_widget import DataTableModel, TableWidget


class TestDataTableModel():

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

    @pytest.mark.xfail
    def test_data_2d(self):
        self.data = Data(x=[[1, 2], [3, 4]], y=[[2, 3], [4, 5]])
        self.model = DataTableModel(self.data)
        for i, c in enumerate(self.data.visible_components):
            for j in range(self.data.size):
                idx = self.model.index(j, i)
                result = self.model.data(idx, Qt.DisplayRole)
                assert float(result) == self.data[c].ravel()[j]


def check_values_and_color(model, data, colors):

    for i in range(len(colors)):

        for j, colname in enumerate('abc'):

            # Get index of cell
            idx = model.index(i, j)

            # Check values
            value = model.data(idx, Qt.DisplayRole)
            assert value == str(data[colname][i])

            # Check colors
            brush = model.data(idx, Qt.BackgroundRole)
            if colors[i] is None:
                assert brush is None
            else:
                assert qt4_to_mpl_color(brush.color()) == colors[i]


def test_table_widget():

    # TODO: add tests for doing the selection interactively

    d = Data(a=[1, 2, 3, 4, 5],
             b=[3.2, 1.2, 4.5, 3.3, 2.2],
             c=['e', 'b', 'c', 'a', 'f'])
    dc = DataCollection([d])
    session = Session(dc, hub=dc.hub)

    widget = TableWidget(session)
    widget.register_to_hub(dc.hub)
    widget.add_data(d)
    widget.show()

    sg1 = dc.new_subset_group('D >= 3', d.id['a'] <= 3)
    sg1.style.color = '#aa0000'
    sg2 = dc.new_subset_group('1 < D < 4', (d.id['a'] > 1) & (d.id['a'] < 4))
    sg2.style.color = '#0000cc'

    model = widget.ui.table.model()

    data = {
        'a': [1, 2, 3, 4, 5],
        'b': [3.2, 1.2, 4.5, 3.3, 2.2],
        'c': ['e', 'b', 'c', 'a', 'f']
    }

    colors = ['#aa0000', '#380088', '#380088', None, None]

    check_values_and_color(model, data, colors)

    model.sort(1, Qt.AscendingOrder)

    data = {
        'a': [2, 5, 1, 4, 3],
        'b': [1.2, 2.2, 3.2, 3.3, 4.5],
        'c': ['b', 'f', 'e', 'a', 'c']
    }

    colors = ['#380088', None, '#aa0000', None, '#380088']

    check_values_and_color(model, data, colors)

    model.sort(2, Qt.AscendingOrder)

    data = {
        'a': [4, 2, 3, 1, 5],
        'b': [3.3, 1.2, 4.5, 3.2, 2.2],
        'c': ['a', 'b', 'c', 'e', 'f']
    }

    colors = [None, '#380088', '#380088', '#aa0000', None]

    check_values_and_color(model, data, colors)

    model.sort(0, Qt.DescendingOrder)

    data = {
        'a': [5, 4, 3, 2, 1],
        'b': [2.2, 3.3, 4.5, 1.2, 3.2],
        'c': ['f', 'a', 'c', 'b', 'e']
    }

    colors = [None, None, '#380088', '#380088', '#aa0000']

    check_values_and_color(model, data, colors)