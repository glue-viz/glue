# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import matplotlib.pyplot as plt
from mock import MagicMock, patch

from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui
from glue.core import Data, Subset
from glue.config import data_factory

from glue.viewers.image.layer_artist import RGBImageLayerArtist

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
