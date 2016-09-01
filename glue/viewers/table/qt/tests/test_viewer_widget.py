from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from qtpy import QtCore, QtGui
from glue.utils.qt import get_qapp
from qtpy.QtCore import Qt
from glue.core import Data, DataCollection, Session
from glue.utils.qt import qt4_to_mpl_color
from glue.app.qt import GlueApplication

from ..viewer_widget import DataTableModel, TableWidget

from glue.core.edit_subset_mode import (EditSubsetMode, AndNotMode, OrMode,
                                        ReplaceMode)


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


def test_table_widget(tmpdir):

    # Start off by creating a glue application instance with a table viewer and
    # some data pre-loaded.

    app = get_qapp()

    d = Data(a=[1, 2, 3, 4, 5],
             b=[3.2, 1.2, 4.5, 3.3, 2.2],
             c=['e', 'b', 'c', 'a', 'f'])

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    widget = gapp.new_data_viewer(TableWidget)
    widget.add_data(d)

    subset_mode = EditSubsetMode()

    # Create two subsets

    sg1 = dc.new_subset_group('D <= 3', d.id['a'] <= 3)
    sg1.style.color = '#aa0000'
    sg2 = dc.new_subset_group('1 < D < 4', (d.id['a'] > 1) & (d.id['a'] < 4))
    sg2.style.color = '#0000cc'

    model = widget.ui.table.model()

    # We now check what the data and colors of the table are, and try various
    # sorting methods to make sure that things are still correct.

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

    model.sort(0, Qt.AscendingOrder)

    # We now modify the subsets using the table.

    selection = widget.ui.table.selectionModel()

    widget.toolbar.actions['table:rowselect'].toggle()

    def press_key(key):
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, Qt.NoModifier)
        app.postEvent(widget.ui.table, event)
        app.processEvents()

    app.processEvents()

    # We now use key presses to navigate down to the third row

    press_key(Qt.Key_Tab)
    press_key(Qt.Key_Down)
    press_key(Qt.Key_Down)

    indices = selection.selectedRows()

    # We make sure that the third row is selected

    assert len(indices) == 1
    assert indices[0].row() == 2

    # At this point, the subsets haven't changed yet

    np.testing.assert_equal(d.subsets[0].to_mask(), [1, 1, 1, 0, 0])
    np.testing.assert_equal(d.subsets[1].to_mask(), [0, 1, 1, 0, 0])

    # We specify that we are editing the second subset, and use a 'not' logical
    # operation to remove the currently selected line from the second subset.

    d.edit_subset = [d.subsets[1]]

    subset_mode.mode = AndNotMode

    press_key(Qt.Key_Enter)

    np.testing.assert_equal(d.subsets[0].to_mask(), [1, 1, 1, 0, 0])
    np.testing.assert_equal(d.subsets[1].to_mask(), [0, 1, 0, 0, 0])

    # At this point, the selection should be cleared

    indices = selection.selectedRows()
    assert len(indices) == 0

    # We move to the fourth row and now do an 'or' selection with the first
    # subset.

    press_key(Qt.Key_Down)

    subset_mode.mode = OrMode

    d.edit_subset = [d.subsets[0]]

    press_key(Qt.Key_Enter)

    np.testing.assert_equal(d.subsets[0].to_mask(), [1, 1, 1, 1, 0])
    np.testing.assert_equal(d.subsets[1].to_mask(), [0, 1, 0, 0, 0])

    # Finally we move to the fifth row and deselect all subsets so that
    # pressing enter now creates a new subset.

    press_key(Qt.Key_Down)

    subset_mode.mode = ReplaceMode

    d.edit_subset = None

    press_key(Qt.Key_Enter)

    np.testing.assert_equal(d.subsets[0].to_mask(), [1, 1, 1, 1, 0])
    np.testing.assert_equal(d.subsets[1].to_mask(), [0, 1, 0, 0, 0])
    np.testing.assert_equal(d.subsets[2].to_mask(), [0, 0, 0, 0, 1])

    # Make the color for the new subset deterministic
    dc.subset_groups[2].style.color = '#bababa'

    # Now finally check saving and restoring session

    session_file = tmpdir.join('table.glu').strpath

    gapp.save_session(session_file)

    gapp2 = GlueApplication.restore_session(session_file)
    gapp2.show()

    d = gapp2.data_collection[0]

    widget2 = gapp2.viewers[0][0]

    model2 = widget2.ui.table.model()

    data = {
        'a': [1, 2, 3, 4, 5],
        'b': [3.2, 1.2, 4.5, 3.3, 2.2],
        'c': ['e', 'b', 'c', 'a', 'f']
    }

    # Need to take into account new selections above
    colors = ['#aa0000', '#380088', '#aa0000', "#aa0000", "#bababa"]

    check_values_and_color(model2, data, colors)
