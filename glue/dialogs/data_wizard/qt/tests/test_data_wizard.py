from __future__ import absolute_import, division, print_function

from mock import MagicMock, patch

from glue.core import Data
from glue.config import data_factory
from ..data_wizard_dialog import GlueDataDialog, data_wizard


@data_factory('testing_factory', identifier=lambda *args: True, priority=-999)
def dummy_factory(filename):
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
            fd._fd.selectNameFilter(v)
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
        assert len(fd.filters) == len([x for x in data_factory.members if not x.deprecated])

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
    with patch('glue.dialogs.data_wizard.qt.data_wizard_dialog.GlueDataDialog') as mock:
        mock().load_data.return_value = []
        assert data_wizard() == []


def test_data_wizard_normal():
    """Returns data list if successful"""
    with patch('glue.dialogs.data_wizard.qt.data_wizard_dialog.GlueDataDialog') as mock:
        mock().load_data.return_value = [1]
        assert data_wizard() == [1]


def test_data_wizard_error_cancel():
    """Returns empty list of error generated and then canceled"""
    with patch('glue.dialogs.data_wizard.qt.data_wizard_dialog.GlueDataDialog') as mock:
        mock().load_data.side_effect = Exception
        with patch('qtpy.QtWidgets.QMessageBox') as qmb:
            qmb().exec_.return_value = 0
            assert data_wizard() == []
