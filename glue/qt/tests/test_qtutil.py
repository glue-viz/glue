#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
from .. import qtutil
from PyQt4 import QtGui
from mock import MagicMock, patch
from ..qtutil import GlueDataDialog


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


def dummy_factory(filename):
    from glue.core import Data
    result = Data()
    result.made_with_dummy_factory = True
    return result

dummy_factory.label = "testing_factory"


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
        assert fd.load_data() is None

    def test_load_data_normal(self):
        """normal load_data dispatches path to factory"""
        fd = GlueDataDialog()
        mock_file_exec(fd, cancel=False, path='ld_data_nrml',
                       factory=dummy_factory)
        d = fd.load_data()
        assert d.label == 'ld_data_nrml'
        assert d.made_with_dummy_factory == True

    def test_filters(self):
        """Should build filter list from data_factories env var"""
        fd = GlueDataDialog()
        import glue
        assert len(fd.filters) == len(glue.env.data_factories)


def mock_file_exec(fd, cancel=False, path='junk', factory=dummy_factory):
    fd._fd.exec_ = MagicMock()
    fd._fd.exec_.return_value = 1 - cancel
    fd.factory = MagicMock()
    fd.factory.return_value = factory
    fd.path = MagicMock()
    fd.path.return_value = path


def test_data_wizard_cancel():
    """Returns empty list if user cancel's dialog"""
    with patch('glue.qt.qtutil.GlueDataDialog') as mock:
        mock().load_data.return_value = None
        assert qtutil.data_wizard() == []


def test_data_wizard_normal():
    """Returns data list if successful"""
    with patch('glue.qt.qtutil.GlueDataDialog') as mock:
        mock().load_data.return_value = 1
        assert qtutil.data_wizard() == [1]


def test_data_wizard_error_cancel():
    """Returns empty list of error generated and then canceled"""
    with patch('glue.qt.qtutil.GlueDataDialog') as mock:
        mock().load_data.side_effect = Exception
        with patch('glue.qt.qtutil.QMessageBox.critical') as critical:
            critical.return_value = 0
            assert qtutil.data_wizard() == []
