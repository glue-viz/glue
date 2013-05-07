#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from .. import qtutil
from ...external.qt import QtGui
from ...external.qt.QtCore import Qt
from mock import MagicMock, patch
from ..qtutil import GlueDataDialog
from ..qtutil import pretty_number, GlueComboBox

from glue.config import data_factory


def test_glue_action_button():
    a = QtGui.QAction(None)
    a.setToolTip("testtooltip")
    a.setWhatsThis("testwhatsthis")
    a.setIcon(QtGui.QIcon("dummy_file"))
    a.setText('testtext')
    b = qtutil.GlueActionButton()
    b.set_action(a)

    #assert b.icon() == a.icon() icons are copied, apparently
    assert b.text() == a.text()
    assert b.toolTip() == a.toolTip()
    assert b.whatsThis() == a.whatsThis()

    #stays in sync
    a.setText('test2')
    assert b.text() == 'test2'


@data_factory('testing_factory', '*.*')
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


def mock_file_exec(fd, cancel=False, path='junk',
                   factory=dummy_factory_member):

    fd._fd.exec_ = MagicMock()
    fd._fd.exec_.return_value = 1 - cancel
    fd.factory = MagicMock()
    fd.factory.return_value = factory
    fd.path = MagicMock()
    fd.path.return_value = path


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

        #Qt swallows exceptions in signals, so we can't assert in this
        #instead, store state and assert after signal
        good = [False]

        def assert_consistent(*args):
            good[0] = len(self.combo._data) == self.combo.count()

        #addItem
        self.combo.currentIndexChanged.connect(assert_consistent)
        self.combo.addItem('a', 1)
        assert good[0]

        #addItems
        self.combo.clear()
        good[0] = False
        self.combo.addItems('b c d'.split())
        assert good[0]

        #removeItem
        self.combo.clear()
        self.combo.addItem('a', 1)
        good[0] = False
        self.combo.removeItem(0)
        assert good[0]
