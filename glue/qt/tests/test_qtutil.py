# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function
import pytest

from .. import qtutil
from ...external.qt import QtGui
from ...external.qt.QtCore import Qt
from mock import MagicMock, patch
from ..qtutil import GlueDataDialog
from ..qtutil import pretty_number, GlueComboBox, PythonListModel

from ...core.config import data_factory
from ...core import Subset


def test_glue_action_button():
    a = QtGui.QAction(None)
    a.setToolTip("testtooltip")
    a.setWhatsThis("testwhatsthis")
    a.setIcon(QtGui.QIcon("dummy_file"))
    a.setText('testtext')
    b = qtutil.GlueActionButton()
    b.set_action(a)

    # assert b.icon() == a.icon() icons are copied, apparently
    assert b.text() == a.text()
    assert b.toolTip() == a.toolTip()
    assert b.whatsThis() == a.whatsThis()

    #stays in sync
    a.setText('test2')
    assert b.text() == 'test2'


@data_factory('testing_factory', identifier=lambda *args: True, priority=-999)
def dummy_factory(filename):
    from glue.core import Data
    result = Data()
    result.made_with_dummy_factory = True
    return result
dummy_factory_member = [f for f in data_factory.members
                        if f[0] is dummy_factory][0]


class TestGlueDataDialog(object):

    def test_factory(self):
        """Factory method should always match with filter"""
        fd = GlueDataDialog()
        assert len(fd.filters) > 0
        for k, v in fd.filters:
            fd._fd.setFilter(v)
            assert fd.factory() is k

    def test_load_data_cancel(self):
        """Return None if user cancels operation"""
        fd = GlueDataDialog()
        mock_file_exec(fd, cancel=True)
        assert fd.load_data() == []

    def test_load_data_normal(self):
        """normal load_data dispatches path to factory"""
        fd = GlueDataDialog()
        mock_file_exec(fd, cancel=False, path='ld_data_nrml',
                       factory=dummy_factory_member)
        d = fd.load_data()
        assert len(d) == 1
        d = d[0]
        assert d.label == 'ld_data_nrml'
        assert d.made_with_dummy_factory is True

    def test_filters(self):
        """Should build filter list from data_factories env var"""
        fd = GlueDataDialog()
        assert len(fd.filters) == len(data_factory.members)

    def test_load_multiple(self):
        fd = GlueDataDialog()
        mock_file_exec(fd, cancel=False, path=['a.fits', 'b.fits'],
                       factory=dummy_factory_member)
        ds = fd.load_data()
        assert len(ds) == 2
        for d, label in zip(ds, 'ab'):
            assert d.label == label
            assert d.made_with_dummy_factory is True


def mock_file_exec(fd, cancel=False, path='junk',
                   factory=dummy_factory_member):
    if not isinstance(path, list):
        path = [path]

    fd._fd.exec_ = MagicMock()
    fd._fd.exec_.return_value = 1 - cancel
    fd.factory = MagicMock()
    fd.factory.return_value = factory
    fd.paths = MagicMock()
    fd.paths.return_value = path


def test_data_wizard_cancel():
    """Returns empty list if user cancel's dialog"""
    with patch('glue.qt.qtutil.GlueDataDialog') as mock:
        mock().load_data.return_value = []
        assert qtutil.data_wizard() == []


def test_data_wizard_normal():
    """Returns data list if successful"""
    with patch('glue.qt.qtutil.GlueDataDialog') as mock:
        mock().load_data.return_value = [1]
        assert qtutil.data_wizard() == [1]


def test_data_wizard_error_cancel():
    """Returns empty list of error generated and then canceled"""
    with patch('glue.qt.qtutil.GlueDataDialog') as mock:
        mock().load_data.side_effect = Exception
        with patch('glue.qt.qtutil.QMessageBox') as qmb:
            qmb().exec_.return_value = 0
            assert qtutil.data_wizard() == []


class TestPrettyNumber(object):

    def test_single(self):
        assert pretty_number([1]) == ['1']
        assert pretty_number([0]) == ['0']
        assert pretty_number([-1]) == ['-1']
        assert pretty_number([1.0001]) == ['1']
        assert pretty_number([1.01]) == ['1.01']
        assert pretty_number([1e-5]) == ['1.000e-05']
        assert pretty_number([1e5]) == ['1.000e+05']
        assert pretty_number([3.3]) == ['3.3']

    def test_list(self):
        assert pretty_number([1, 2, 3.3, 1e5]) == ['1', '2', '3.3',
                                                   '1.000e+05']


class TestGlueComboBox(object):

    def setup_method(self, method):
        self.combo = GlueComboBox()

    def test_add_data(self):
        self.combo.addItem('hi', userData=3)
        assert self.combo.itemData(0) == 3

    def test_add_multi_data(self):
        self.combo.addItem('hi', userData=3)
        self.combo.addItem('ho', userData=4)
        assert self.combo.itemData(0) == 3
        assert self.combo.itemData(1) == 4

    def test_replace(self):
        self.combo.addItem('hi', userData=3)
        self.combo.removeItem(0)
        self.combo.addItem('ho', userData=4)
        assert self.combo.itemData(0) == 4

    def test_clear(self):
        self.combo.addItem('a', 1)
        self.combo.addItem('b', 2)
        self.combo.addItem('c', 3)
        self.combo.clear()
        self.combo.addItem('d', 4)
        assert self.combo.itemData(0) == 4

    def test_mid_remove(self):
        self.combo.addItem('a', 1)
        self.combo.addItem('b', 2)
        self.combo.addItem('c', 3)
        self.combo.removeItem(1)
        assert self.combo.itemData(1) == 3

    def test_set_item_data(self):
        self.combo.addItem('a', 1)
        self.combo.setItemData(0, 2)
        assert self.combo.itemData(0) == 2

    def test_default_data(self):
        self.combo.addItem('a')
        assert self.combo.itemData(0) is None

    def test_add_items(self):
        self.combo.addItem('a', 1)
        self.combo.addItems(['b', 'c', 'd'])
        assert self.combo.itemData(0) == 1
        assert self.combo.itemData(1) is None
        assert self.combo.itemData(2) is None
        assert self.combo.itemData(3) is None

    def test_non_user_role(self):
        """methods that edit data other than userRole dispatched to super"""
        self.combo.addItem('a', 1)
        assert self.combo.itemData(0, role=Qt.DisplayRole) == 'a'
        self.combo.setItemData(0, 'b', role=Qt.DisplayRole)
        assert self.combo.itemData(0, role=Qt.DisplayRole) == 'b'

    def test_consistent_with_signals(self):
        """Ensure that when signal/slot connections interrupt
        methods mid-call, internal data state is consistent"""

        # Qt swallows exceptions in signals, so we can't assert in this
        # instead, store state and assert after signal
        good = [False]

        def assert_consistent(*args):
            good[0] = len(self.combo._data) == self.combo.count()

        # addItem
        self.combo.currentIndexChanged.connect(assert_consistent)
        self.combo.addItem('a', 1)
        assert good[0]

        # addItems
        self.combo.clear()
        good[0] = False
        self.combo.addItems('b c d'.split())
        assert good[0]

        # removeItem
        self.combo.clear()
        self.combo.addItem('a', 1)
        good[0] = False
        self.combo.removeItem(0)
        assert good[0]


def test_qt4_to_mpl_color():
    assert qtutil.qt4_to_mpl_color(QtGui.QColor(255, 0, 0)) == '#ff0000'
    assert qtutil.qt4_to_mpl_color(QtGui.QColor(255, 255, 255)) == '#ffffff'


def test_edit_color():
    with patch('glue.qt.qtutil.QColorDialog') as d:
        d.getColor.return_value = QtGui.QColor(0, 1, 0)
        d.isValid.return_value = True
        s = Subset(None)
        qtutil.edit_layer_color(s)
        assert s.style.color == '#000100'


def test_edit_color_cancel():
    with patch('glue.qt.qtutil.QColorDialog') as d:
        d.getColor.return_value = QtGui.QColor(0, -1, 0)
        s = Subset(None)
        qtutil.edit_layer_color(s)


def test_edit_symbol():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getItem.return_value = ('*', True)
        s = Subset(None)
        qtutil.edit_layer_symbol(s)
        assert s.style.marker == '*'


def test_edit_symbol_cancel():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getItem.return_value = ('*', False)
        s = Subset(None)
        qtutil.edit_layer_symbol(s)
        assert s.style.marker != '*'


def test_edit_point_size():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getInt.return_value = 123, True
        s = Subset(None)
        qtutil.edit_layer_point_size(s)
        assert s.style.markersize == 123


def test_edit_point_size_cancel():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getInt.return_value = 123, False
        s = Subset(None)
        qtutil.edit_layer_point_size(s)
        assert s.style.markersize != 123


def test_edit_layer_label():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getText.return_value = ('accepted label', True)
        s = Subset(None)
        qtutil.edit_layer_label(s)
        assert s.label == 'accepted label'


def test_edit_layer_label_cancel():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getText.return_value = ('rejected label', False)
        s = Subset(None)
        qtutil.edit_layer_label(s)
        assert s.label != 'rejected label'


def test_pick_item():
    items = ['a', 'b', 'c']
    labels = ['1', '2', '3']
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getItem.return_value = '1', True
        assert qtutil.pick_item(items, labels) == 'a'
        d.getItem.return_value = '2', True
        assert qtutil.pick_item(items, labels) == 'b'
        d.getItem.return_value = '3', True
        assert qtutil.pick_item(items, labels) == 'c'
        d.getItem.return_value = '3', False
        assert qtutil.pick_item(items, labels) is None


def test_pick_class():
    class Foo:
        pass

    class Bar:
        pass
    Bar.LABEL = 'Baz'
    with patch('glue.qt.qtutil.pick_item') as d:
        qtutil.pick_class([Foo, Bar])
        d.assert_called_once_with([Foo, Bar], ['Foo', 'Baz'])


def test_get_text():
    with patch('glue.qt.qtutil.QInputDialog') as d:
        d.getText.return_value = 'abc', True
        assert qtutil.get_text() == 'abc'

        d.getText.return_value = 'abc', False
        assert qtutil.get_text() is None


class TestGlueListWidget(object):

    def setup_method(self, method):
        self.w = qtutil.GlueListWidget()

    def test_mime_type(self):
        assert self.w.mimeTypes() == [qtutil.LAYERS_MIME_TYPE]

    def test_mime_data(self):
        self.w.set_data(3, 'test data')
        self.w.set_data(4, 'do not pick')
        mime = self.w.mimeData([3])
        mime.data(qtutil.LAYERS_MIME_TYPE) == ['test data']

    def test_mime_data_multiselect(self):
        self.w.set_data(3, 'test data')
        self.w.set_data(4, 'also pick')
        mime = self.w.mimeData([3, 4])
        mime.data(qtutil.LAYERS_MIME_TYPE) == ['test data', 'also pick']


class TestRGBEdit(object):

    def setup_method(self, method):
        from glue.clients.layer_artist import RGBImageLayerArtist
        from glue.core import Data
        d = Data()
        self.artist = RGBImageLayerArtist(d, None)
        self.w = qtutil.RGBEdit(artist=self.artist)

    def test_update_visible(self):
        for color in ['red', 'green', 'blue']:
            state = self.artist.layer_visible[color]
            self.w.vis[color].click()
            assert self.artist.layer_visible[color] != state

    def test_update_current(self):
        for color in ['red', 'green', 'blue']:
            self.w.current[color].click()
            assert self.artist.contrast_layer == color


class TestListModel(object):

    def test_row_count(self):
        assert PythonListModel([]).rowCount() == 0
        assert PythonListModel([1]).rowCount() == 1
        assert PythonListModel([1, 2]).rowCount() == 2

    def test_data_display(self):
        m = PythonListModel([1, 'a'])
        i = m.index(0)
        assert m.data(i, role=Qt.DisplayRole) == '1'

        i = m.index(1)
        assert m.data(i, role=Qt.DisplayRole) == 'a'

    def test_data_edit(self):
        m = PythonListModel([1, 'a'])
        i = m.index(0)
        assert m.data(i, role=Qt.EditRole) == '1'

        i = m.index(1)
        assert m.data(i, role=Qt.EditRole) == 'a'

    def test_data_user(self):
        m = PythonListModel([1, 'a'])
        i = m.index(0)
        assert m.data(i, role=Qt.UserRole) == 1

        i = m.index(1)
        assert m.data(i, role=Qt.UserRole) == 'a'

    def test_itemget(self):
        m = PythonListModel([1, 'a'])
        assert m[0] == 1
        assert m[1] == 'a'

    def test_itemset(self):
        m = PythonListModel([1, 'a'])
        m[0] = 'b'
        assert m[0] == 'b'

    @pytest.mark.parametrize('items', ([], [1, 2, 3], [1]))
    def test_len(self, items):
        assert len(PythonListModel(items)) == len(items)

    def test_pop(self):
        m = PythonListModel([1, 2, 3])
        assert m.pop() == 3
        assert len(m) == 2
        assert m.pop(0) == 1
        assert len(m) == 1
        assert m[0] == 2

    def test_append(self):
        m = PythonListModel([])
        m.append(2)
        assert m[0] == 2
        m.append(3)
        assert m[1] == 3
        m.pop()
        m.append(4)
        assert m[1] == 4

    def test_extend(self):
        m = PythonListModel([])
        m.extend([2, 3])
        assert m[0] == 2
        assert m[1] == 3

    def test_insert(self):
        m = PythonListModel([1, 2, 3])
        m.insert(1, 5)
        assert m[1] == 5

    def test_iter(self):
        m = PythonListModel([1, 2, 3])
        assert list(m) == [1, 2, 3]
