# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

from mock import patch

from glue.external.qt import QtGui
from glue.core import Subset

from .. import qtutil


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
