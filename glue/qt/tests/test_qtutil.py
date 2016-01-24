# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import matplotlib.pyplot as plt
from mock import MagicMock, patch

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui
from glue.core import Data, Subset
from glue.config import data_factory
from glue.clients.layer_artist import RGBImageLayerArtist

from .. import qtutil
from ..qtutil import GlueDataDialog

from ...utils.array import pretty_number
from ...utils.qt import PythonListModel, update_combobox


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



def test_edit_color():
    with patch('glue.qt.qtutil.QtGui.QColorDialog') as d:
        d.getColor.return_value = QtGui.QColor(0, 1, 0)
        d.isValid.return_value = True
        s = Subset(None)
        qtutil.edit_layer_color(s)
        assert s.style.color == '#000100'


def test_edit_color_cancel():
    with patch('glue.qt.qtutil.QtGui.QColorDialog') as d:
        d.getColor.return_value = QtGui.QColor(0, -1, 0)
        s = Subset(None)
        qtutil.edit_layer_color(s)


def test_edit_symbol():
    with patch('glue.qt.qtutil.QtGui.QInputDialog') as d:
        d.getItem.return_value = ('*', True)
        s = Subset(None)
        qtutil.edit_layer_symbol(s)
        assert s.style.marker == '*'


def test_edit_symbol_cancel():
    with patch('glue.qt.qtutil.QtGui.QInputDialog') as d:
        d.getItem.return_value = ('*', False)
        s = Subset(None)
        qtutil.edit_layer_symbol(s)
        assert s.style.marker != '*'


def test_edit_point_size():
    with patch('glue.qt.qtutil.QtGui.QInputDialog') as d:
        d.getInt.return_value = 123, True
        s = Subset(None)
        qtutil.edit_layer_point_size(s)
        assert s.style.markersize == 123


def test_edit_point_size_cancel():
    with patch('glue.qt.qtutil.QtGui.QInputDialog') as d:
        d.getInt.return_value = 123, False
        s = Subset(None)
        qtutil.edit_layer_point_size(s)
        assert s.style.markersize != 123


def test_edit_layer_label():
    with patch('glue.qt.qtutil.QtGui.QInputDialog') as d:
        d.getText.return_value = ('accepted label', True)
        s = Subset(None)
        qtutil.edit_layer_label(s)
        assert s.label == 'accepted label'


def test_edit_layer_label_cancel():
    with patch('glue.qt.qtutil.QtGui.QInputDialog') as d:
        d.getText.return_value = ('rejected label', False)
        s = Subset(None)
        qtutil.edit_layer_label(s)
        assert s.label != 'rejected label'


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
        d = Data()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.artist = RGBImageLayerArtist(d, self.ax)
        self.w = qtutil.RGBEdit(artist=self.artist)

    def teardown_method(self, method):
        plt.close(self.fig)

    def test_update_visible(self):
        for color in ['red', 'green', 'blue']:
            state = self.artist.layer_visible[color]
            self.w.vis[color].click()
            assert self.artist.layer_visible[color] != state

    def test_update_current(self):
        for color in ['red', 'green', 'blue']:
            self.w.current[color].click()
            assert self.artist.contrast_layer == color


