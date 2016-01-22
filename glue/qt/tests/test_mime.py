from __future__ import absolute_import, division, print_function

import pytest

from glue.external.qt.QtTest import QTest
from glue.external.qt.QtCore import Qt
from glue.external.qt import QtGui

from .. import mime


class TestWidget(QtGui.QWidget):
    def __init__(self, out_mime, parent=None):
        super(TestWidget, self).__init__(parent)
        self.setAcceptDrops(True)

        self.last_mime = None
        self.out_mime = out_mime

    def dragEnterEvent(self, event):
        print('drag enter')
        event.accept()

    def dropEvent(self, event):
        print('drop')
        self.last_mime = event.mimeData()

    def mousePressEvent(self, event):
        print('mouse event')
        drag = QtGui.QDrag(self)
        drag.setMimeData(self.out_mime)
        drop_action = drag.exec_()
        print(drop_action)
        event.accept()


class TestMime(object):
    def test_formats(self):
        d = mime.PyMimeData()
        assert set(d.formats()) == set([mime.INSTANCE_MIME_TYPE])

        d = mime.PyMimeData(**{'text/plain': 'hello'})
        assert set(d.formats()) == set([mime.INSTANCE_MIME_TYPE, 'text/plain'])

    def test_empty_has_format(self):
        d = mime.PyMimeData()
        assert d.hasFormat(mime.INSTANCE_MIME_TYPE)
        assert not d.hasFormat(mime.LAYER_MIME_TYPE)
        assert not d.hasFormat(mime.LAYERS_MIME_TYPE)

    def test_instance_format(self):
        d = mime.PyMimeData(5)
        assert d.hasFormat(mime.INSTANCE_MIME_TYPE)
        assert not d.hasFormat(mime.LAYER_MIME_TYPE)
        assert not d.hasFormat(mime.LAYERS_MIME_TYPE)

    def test_layer_format(self):
        d = mime.PyMimeData(5, **{mime.LAYER_MIME_TYPE: 10})
        assert d.hasFormat(mime.INSTANCE_MIME_TYPE)
        assert d.hasFormat(mime.LAYER_MIME_TYPE)
        assert not d.hasFormat(mime.LAYERS_MIME_TYPE)

    def test_layers_format(self):
        d = mime.PyMimeData(5, **{mime.LAYERS_MIME_TYPE: 10})
        assert d.hasFormat(mime.INSTANCE_MIME_TYPE)
        assert d.hasFormat(mime.LAYERS_MIME_TYPE)
        assert not d.hasFormat(mime.LAYER_MIME_TYPE)

    def test_retrieve_instance(self):
        d = mime.PyMimeData(10)
        assert d.retrieveData(mime.INSTANCE_MIME_TYPE) == 10

    def test_retrieve_layer(self):
        d = mime.PyMimeData(**{mime.LAYERS_MIME_TYPE: 12})
        assert d.retrieveData(mime.LAYERS_MIME_TYPE) == 12

        d = mime.PyMimeData(**{mime.LAYER_MIME_TYPE: 12})
        assert d.retrieveData(mime.LAYER_MIME_TYPE) == 12

    def test_retrieve_not_present_returns_null(self):
        d = mime.PyMimeData()
        assert d.retrieveData('not-a-format') is None


@pytest.skip
class TestMimeDragAndDrop(object):
    def setup_method(self, method):

        m1 = mime.PyMimeData(1, **{'text/plain': 'hi', 'test': 4})
        m2 = mime.PyMimeData(1, **{'test': 5})

        w1 = TestWidget(m1)
        w2 = TestWidget(m2)

        self.w1 = w1
        self.w2 = w2
        self.m1 = m1
        self.m2 = m2

    def test_drag_drop(self):
        QTest.mousePress(self.w1, Qt.LeftButton)
        QTest.mouseMove(self.w2)
        QTest.mouseRelease(self.w2, Qt.LeftButton)

        assert self.w2.last_mime == self.m1
