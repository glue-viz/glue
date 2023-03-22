import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt

from glue.utils.qt import get_qapp, process_events
from glue.core import Data, DataCollection
from glue.utils.qt import qt_to_mpl_color
from glue.app.qt import GlueApplication

from ..data_viewer import DataTableModel, TableViewer

from glue.core.edit_subset_mode import AndNotMode, OrMode, ReplaceMode
from glue.tests.helpers import requires_pyqt_gt_59_or_pyside2


class TestDataTableModel():

    def setup_method(self, method):
        self.gapp = GlueApplication()
        self.viewer = self.gapp.new_data_viewer(TableViewer)
        self.data = Data(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        self.gapp.data_collection.append(self.data)
        self.viewer.add_data(self.data)
        self.model = DataTableModel(self.viewer)

    def teardown_method(self, method):
        self.gapp.close()
        self.gapp = None

    def test_column_count(self):
        assert self.model.columnCount() == 2

    def test_column_count_hidden(self):
        self.model.show_coords = True
        assert self.model.columnCount() == 3

    def test_header_data(self):
        for i, c in enumerate(self.data.main_components):
            result = self.model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
            assert result == c.label

        for i in range(self.data.size):
            result = self.model.headerData(i, Qt.Vertical, Qt.DisplayRole)
            assert result == str(i)

    def test_row_count(self):
        assert self.model.rowCount() == 4

    def test_data(self):
        for i, c in enumerate(self.data.main_components):
            for j in range(self.data.size):
                idx = self.model.index(j, i)
                result = self.model.data(idx, Qt.DisplayRole)
                assert float(result) == self.data[c, j]

    @pytest.mark.xfail
    def test_data_2d(self):
        self.data = Data(x=[[1, 2], [3, 4]], y=[[2, 3], [4, 5]])
        self.model = DataTableModel(self.data)
        for i, c in enumerate(self.data.main_components):
            for j in range(self.data.size):
                idx = self.model.index(j, i)
                result = self.model.data(idx, Qt.DisplayRole)
                assert float(result) == self.data[c].ravel()[j]


def check_values_and_color(model, data, colors):

    for i in range(len(colors)):

        for j, colname in enumerate(sorted(data)):

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
                assert qt_to_mpl_color(brush.color()) == colors[i]


def test_table_widget(tmpdir):

    # Start off by creating a glue application instance with a table viewer and
    # some data pre-loaded.

    app = get_qapp()

    d = Data(a=[1, 2, 3, 4, 5],
             b=[3.2, 1.2, 4.5, 3.3, 2.2],
             c=['e', 'b', 'c', 'a', 'f'])

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    widget = gapp.new_data_viewer(TableViewer)
    widget.add_data(d)

    subset_mode = gapp._session.edit_subset_mode

    # Create two subsets

    sg1 = dc.new_subset_group('D <= 3', d.id['a'] <= 3)
    sg1.style.color = '#aa0000'
    sg2 = dc.new_subset_group('1 < D < 4', (d.id['a'] > 1) & (d.id['a'] < 4))
    sg2.style.color = '#0000cc'

    model = widget.ui.table.model()

    # We now check what the data and colors of the table are, and try various
    # sorting methods to make sure that things are still correct.

    data = {'a': [1, 2, 3, 4, 5],
            'b': [3.2, 1.2, 4.5, 3.3, 2.2],
            'c': ['e', 'b', 'c', 'a', 'f']}

    colors = ['#aa0000', '#380088', '#380088', None, None]

    check_values_and_color(model, data, colors)

    model.sort(1, Qt.AscendingOrder)

    data = {'a': [2, 5, 1, 4, 3],
            'b': [1.2, 2.2, 3.2, 3.3, 4.5],
            'c': ['b', 'f', 'e', 'a', 'c']}

    colors = ['#380088', None, '#aa0000', None, '#380088']

    check_values_and_color(model, data, colors)

    model.sort(2, Qt.AscendingOrder)

    data = {'a': [4, 2, 3, 1, 5],
            'b': [3.3, 1.2, 4.5, 3.2, 2.2],
            'c': ['a', 'b', 'c', 'e', 'f']}

    colors = [None, '#380088', '#380088', '#aa0000', None]

    check_values_and_color(model, data, colors)

    model.sort(0, Qt.DescendingOrder)

    data = {'a': [5, 4, 3, 2, 1],
            'b': [2.2, 3.3, 4.5, 1.2, 3.2],
            'c': ['f', 'a', 'c', 'b', 'e']}

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

    process_events()

    # We now use key presses to navigate down to the third row

    press_key(Qt.Key_Tab)
    press_key(Qt.Key_Down)
    press_key(Qt.Key_Down)

    process_events()

    indices = selection.selectedRows()

    # We make sure that the third row is selected

    assert len(indices) == 1
    assert indices[0].row() == 2

    # At this point, the subsets haven't changed yet

    np.testing.assert_equal(d.subsets[0].to_mask(), [1, 1, 1, 0, 0])
    np.testing.assert_equal(d.subsets[1].to_mask(), [0, 1, 1, 0, 0])

    # We specify that we are editing the second subset, and use a 'not' logical
    # operation to remove the currently selected line from the second subset.

    subset_mode.edit_subset = [d.subsets[1]]
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

    subset_mode.edit_subset = [d.subsets[0]]

    press_key(Qt.Key_Enter)

    np.testing.assert_equal(d.subsets[0].to_mask(), [1, 1, 1, 1, 0])
    np.testing.assert_equal(d.subsets[1].to_mask(), [0, 1, 0, 0, 0])

    # Finally we move to the fifth row and deselect all subsets so that
    # pressing enter now creates a new subset.

    press_key(Qt.Key_Down)

    subset_mode.mode = ReplaceMode

    subset_mode.edit_subset = None

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

    data = {'a': [1, 2, 3, 4, 5],
            'b': [3.2, 1.2, 4.5, 3.3, 2.2],
            'c': ['e', 'b', 'c', 'a', 'f']}

    # Need to take into account new selections above
    colors = ['#aa0000', '#380088', '#aa0000', "#aa0000", "#bababa"]

    check_values_and_color(model2, data, colors)


def test_table_widget_session_no_subset(tmpdir):

    # Regression test for a bug that caused table viewers with no subsets to
    # not be restored correctly and instead raise an exception.

    app = get_qapp()  # noqa

    d = Data(a=[1, 2, 3, 4, 5],
             b=[3.2, 1.2, 4.5, 3.3, 2.2],
             c=['e', 'b', 'c', 'a', 'f'], label='test')

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    widget = gapp.new_data_viewer(TableViewer)
    widget.add_data(d)

    session_file = tmpdir.join('table.glu').strpath

    gapp.save_session(session_file)

    gapp2 = GlueApplication.restore_session(session_file)
    gapp2.show()

    gapp2.data_collection[0]
    gapp2.viewers[0][0]


def test_change_components():

    # Regression test for a bug that caused table viewers to not update when
    # adding/removing components.

    app = get_qapp()  # noqa

    d = Data(a=[1, 2, 3, 4, 5],
             b=[3.2, 1.2, 4.5, 3.3, 2.2],
             c=['e', 'b', 'c', 'a', 'f'], label='test')

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    viewer = gapp.new_data_viewer(TableViewer)
    viewer.add_data(d)

    data_changed = MagicMock()
    viewer.model.dataChanged.connect(data_changed)

    # layoutChanged needs to be emitted for the new/removed columns to be
    # registered (dataChanged is not enough)
    layout_changed = MagicMock()
    viewer.model.layoutChanged.connect(layout_changed)

    assert data_changed.call_count == 0
    assert layout_changed.call_count == 0
    viewer.model.columnCount() == 2

    d.add_component([3, 4, 5, 6, 2], 'z')

    assert data_changed.call_count == 1
    assert layout_changed.call_count == 1
    viewer.model.columnCount() == 3

    d.remove_component(d.id['z'])

    assert data_changed.call_count == 2
    assert layout_changed.call_count == 2
    viewer.model.columnCount() == 2


def test_table_title():

    app = get_qapp()  # noqa

    data1 = Data(a=[1, 2, 3, 4, 5], label='test1')
    data2 = Data(a=[1, 2, 3, 4, 5], label='test2')

    dc = DataCollection([data1, data2])

    gapp = GlueApplication(dc)

    viewer = gapp.new_data_viewer(TableViewer)

    assert viewer.windowTitle() == 'Table'

    viewer.add_data(data1)

    assert viewer.windowTitle() == 'Table: test1'

    viewer.add_data(data2)

    assert viewer.windowTitle() == 'Table: test2'


def test_add_subset():

    # Regression test for a bug that occurred when adding a subset
    # directly to the table viewer.

    data1 = Data(a=[1, 2, 3, 4, 5], label='test1')
    data2 = Data(a=[1, 2, 3, 4, 5], label='test2')
    dc = DataCollection([data1, data2])
    dc.new_subset_group('test subset 1', data1.id['a'] > 2)

    gapp = GlueApplication(dc)

    viewer = gapp.new_data_viewer(TableViewer)
    viewer.add_subset(data1.subsets[0])

    assert len(viewer.state.layers) == 2
    assert not viewer.state.layers[0].visible
    assert viewer.state.layers[1].visible

    dc.new_subset_group('test subset 2', data1.id['a'] <= 2)

    assert len(viewer.state.layers) == 3
    assert not viewer.state.layers[0].visible
    assert viewer.state.layers[1].visible
    assert viewer.state.layers[2].visible

    viewer.remove_subset(data1.subsets[1])

    assert len(viewer.state.layers) == 2
    assert not viewer.state.layers[0].visible
    assert viewer.state.layers[1].visible

    viewer.add_subset(data1.subsets[1])

    assert len(viewer.state.layers) == 3
    assert not viewer.state.layers[0].visible
    assert viewer.state.layers[1].visible
    assert viewer.state.layers[2].visible

    with pytest.raises(ValueError) as exc:
        viewer.add_subset(data2.subsets[1])
    assert exc.value.args[0] == 'subset parent data does not match existing table data'


def test_graceful_close_after_invalid(capsys):

    # Regression test for a bug that caused an error if an invalid dataset
    # was added to the viewer after the user had acknowledged the error.

    d = Data(a=[[1, 2], [3, 4]], label='test')

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    viewer = gapp.new_data_viewer(TableViewer)
    gapp.show()

    process_events()

    with pytest.raises(ValueError, match='Can only use Table widget for 1D data'):
        viewer.add_data(d)

    viewer.close()

    process_events()

    #  We use capsys here because the # error is otherwise only apparent in stderr.
    out, err = capsys.readouterr()
    assert out.strip() == ""
    assert err.strip() == ""


def test_incompatible_subset():

    # Regression test for a bug that caused the table to be refreshed in an
    # infinite loop if incompatible subsets were present.

    data1 = Data(a=[1, 2, 3, 4, 5], label='test1')
    data2 = Data(a=[1, 2, 3, 4, 5], label='test2')
    dc = DataCollection([data1, data2])

    gapp = GlueApplication(dc)

    viewer = gapp.new_data_viewer(TableViewer)
    viewer.add_data(data1)

    dc.new_subset_group('test subset', data2.id['a'] > 2)
    gapp.show()
    process_events(0.5)

    with patch.object(viewer.layers[0], '_refresh') as refresh1:
        with patch.object(viewer.layers[1], '_refresh') as refresh2:
            process_events(0.5)

    assert refresh1.call_count == 0
    assert refresh2.call_count == 0


@requires_pyqt_gt_59_or_pyside2
def test_table_incompatible_attribute():
    """
    Regression test for a bug where the table viewer generates an
    uncaught IncompatibleAttribute error in _update_visible() if
    the dataset is not visible and an invalid subset exists at all.
    This occurred because layer_artists depending on
    invalid attributes were only being disabled (but were still
    visible) and the viewer attempts to compute a mask for
    all visible subsets if the underlying dataset is not visible.
    """
    app = get_qapp()
    d1 = Data(x=[1, 2, 3, 4], y=[5, 6, 7, 8], label='d1')
    d2 = Data(a=['a', 'b', 'c'], b=['x', 'y', 'z'], label='d2')
    dc = DataCollection([d1, d2])
    gapp = GlueApplication(dc)
    viewer = gapp.new_data_viewer(TableViewer)
    viewer.add_data(d2)

    # This subset should not be shown in the viewer
    sg1 = dc.new_subset_group('invalid', d1.id['x'] <= 3)

    gapp.show()
    process_events()

    assert len(viewer.layers) == 2
    assert not viewer.layers[1].visible
    assert viewer.layers[0].visible

    # This subset can be shown in the viewer
    sg2 = dc.new_subset_group('valid', d2.id['a'] == 'a')

    assert len(viewer.layers) == 3
    assert viewer.layers[2].visible
    assert not viewer.layers[1].visible
    assert viewer.layers[0].visible

    # The original IncompatibleAttribute was thrown
    # here as making the data layer invisible made
    # DataTableModel._update_visible() try and draw
    # the invalid subset
    viewer.layers[0].visible = False
    assert viewer.layers[2].visible
    assert not viewer.layers[1].visible


def test_table_with_dask_column():

    da = pytest.importorskip('dask.array')

    app = get_qapp()

    d = Data(d=da.asarray([1, 2, 3, 4, 5]),
             e=np.arange(5) + 1)

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    widget = gapp.new_data_viewer(TableViewer)
    widget.add_data(d)

    sg1 = dc.new_subset_group('D <= 3', d.id['d'] <= 3)
    sg1.style.color = '#aa0000'
    sg2 = dc.new_subset_group('1 < E < 4', (d.id['e'] > 1) & (d.id['e'] < 4))
    sg2.style.color = '#0000cc'

    assert widget.state.layers[0].visible
    assert widget.state.layers[1].visible
    assert widget.state.layers[2].visible

    model = widget.ui.table.model()

    # We now check what the data and colors of the table are, and try various
    # sorting methods to make sure that things are still correct.

    data = {'d': [1, 2, 3, 4, 5],
            'e': [1, 2, 3, 4, 5]}

    colors = ['#aa0000', '#380088', '#380088', None, None]

    check_values_and_color(model, data, colors)

    widget.state.layers[2].visible = False

    colors = ['#aa0000', '#aa0000', '#aa0000', None, None]

    check_values_and_color(model, data, colors)


def test_table_preserve_model_after_selection():

    # Regression test for a bug that caused table viewers to return
    # to default sorting after a new subset was created with the row
    # selection tool. This occurred because the model was reset.

    app = get_qapp()  # noqa

    d = Data(a=[1, 2, 3, 4, 5],
             b=[3.2, 1.2, 4.5, 3.3, 2.2],
             c=['e', 'b', 'c', 'a', 'f'], label='test')

    dc = DataCollection([d])

    gapp = GlueApplication(dc)

    viewer = gapp.new_data_viewer(TableViewer)
    viewer.add_data(d)

    model = viewer.ui.table.model()

    model.sort(1, Qt.AscendingOrder)

    data = {'a': [2, 5, 1, 4, 3],
            'b': [1.2, 2.2, 3.2, 3.3, 4.5],
            'c': ['b', 'f', 'e', 'a', 'c']}
    colors = [None for _ in range(5)]

    check_values_and_color(model, data, colors)

    # Create a new subset using the row selection tool

    subset_mode = gapp._session.edit_subset_mode
    subset_mode.edit_subset = None
    viewer.toolbar.actions['table:rowselect'].toggle()

    def press_key(key):
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, Qt.NoModifier)
        app.postEvent(viewer.ui.table, event)
        app.processEvents()

    # Select the second row
    press_key(Qt.Key_Tab)
    press_key(Qt.Key_Down)
    press_key(Qt.Key_Enter)

    process_events()

    # Check that the table model is still the same, which it
    # should be since we aren't changing the viewer Data

    post_model = viewer.ui.table.model()
    assert post_model == model

    # Check that the order is still the same

    color = d.subsets[0].style.color
    colors[1] = color

    check_values_and_color(post_model, data, colors)
