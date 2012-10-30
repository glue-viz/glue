from ..widget_properties import (ButtonProperty,
                                 FloatLineProperty)

from PyQt4.QtGui import QCheckBox, QLineEdit


class TestClass(object):
    b = ButtonProperty('_button')
    fl = FloatLineProperty('_float')

    def __init__(self):
        self._button = QCheckBox()
        self._float = QLineEdit()


def test_button():
    tc = TestClass()
    assert tc.b == tc._button.checkState()

    tc.b = True
    assert tc._button.isChecked()

    tc.b = False
    assert not tc._button.isChecked()

    tc._button.setChecked(True)
    assert tc.b

    tc._button.setChecked(False)
    assert not tc.b


def test_float():
    tc = TestClass()

    tc.fl = 1.0
    assert float(tc._float.text()) == 1.0

    tc._float.setText('10')
    assert tc.fl == 10.0
