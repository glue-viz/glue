from __future__ import absolute_import, division, print_function

import pytest

from ..widget_properties import (CurrentComboDataProperty,
                                 CurrentComboTextProperty,
                                 CurrentTabProperty,
                                 TextProperty,
                                 ButtonProperty,
                                 FloatLineProperty,
                                 ValueProperty,
                                 connect_bool_button,
                                 connect_current_combo,
                                 connect_float_edit,
                                 connect_int_spin)

from ...external.qt.QtGui import (QCheckBox,
                                  QLineEdit,
                                  QComboBox,
                                  QLabel,
                                  QSlider,
                                  QTabWidget,
                                  QWidget,
                                  QKeyEvent)

from ...external.qt import QtCore, get_qapp

from ...external.echo import CallbackProperty



def test_combo_data():

    class TestClass(object):

        co = CurrentComboDataProperty('_combo')

        def __init__(self):
            self._combo = QComboBox()
            self._combo.addItem('a', 'a')
            self._combo.addItem('b', 'b')

    tc = TestClass()

    tc.co = 'a'
    assert tc.co == 'a'
    assert tc._combo.currentIndex() == 0

    tc.co = 'b'
    assert tc.co == 'b'
    assert tc._combo.currentIndex() == 1

    with pytest.raises(ValueError) as exc:
        tc.co = 'c'
    assert exc.value.args[0] == "Cannot find data 'c' in combo box"


def test_combo_text():

    class TestClass(object):

        co = CurrentComboTextProperty('_combo')

        def __init__(self):
            self._combo = QComboBox()
            self._combo.addItem('a')
            self._combo.addItem('b')

    tc = TestClass()

    tc.co = 'a'
    assert tc.co == 'a'
    assert tc._combo.currentIndex() == 0

    tc.co = 'b'
    assert tc.co == 'b'
    assert tc._combo.currentIndex() == 1

    with pytest.raises(ValueError) as exc:
        tc.co = 'c'
    assert exc.value.args[0] == "Cannot find text 'c' in combo box"


def test_text():

    class TestClass(object):
        lab = TextProperty('_label')
        def __init__(self):
            self._label = QLabel()

    tc = TestClass()
    tc.lab = 'hello'
    assert tc.lab == 'hello'
    assert tc._label.text() == 'hello'


def test_button():

    class TestClass(object):
        but = ButtonProperty('_button')
        def __init__(self):
            self._button = QCheckBox()

    tc = TestClass()

    assert tc.but == tc._button.checkState()

    tc.but = True
    assert tc._button.isChecked()

    tc.but = False
    assert not tc._button.isChecked()

    tc._button.setChecked(True)
    assert tc.but

    tc._button.setChecked(False)
    assert not tc.but


def test_float():

    class TestClass(object):
        flt = FloatLineProperty('_float')
        def __init__(self):
            self._float = QLineEdit()

    tc = TestClass()

    tc.flt = 1.0
    assert float(tc._float.text()) == 1.0

    tc._float.setText('10')
    assert tc.flt == 10.0

    tc._float.setText('')
    assert tc.flt == 0.0


def test_value():

    class TestClass(object):
        val = ValueProperty('_slider')
        def __init__(self):
            self._slider = QSlider()

    tc = TestClass()

    tc.val = 2.0
    assert tc.val == 2.0
    assert tc._slider.value() == 2.0


def test_value_mapping():

    class TestClass(object):
        val = ValueProperty('_slider', mapping=(lambda x: 2 * x,
                                                lambda x: 0.5 * x))
        def __init__(self):
            self._slider = QSlider()

    tc = TestClass()

    tc.val = 2.0
    assert tc.val == 2.0
    assert tc._slider.value() == 1.0


def test_tab():

    class TestClass(object):
        tab = CurrentTabProperty('_tab')
        def __init__(self):
            self._tab = QTabWidget()
            self._tab.addTab(QWidget(), 'tab1')
            self._tab.addTab(QWidget(), 'tab2')

    tc = TestClass()

    tc.tab = 'tab1'
    assert tc.tab == 'tab1'
    assert tc._tab.currentIndex() == 0

    tc.tab = 'tab2'
    assert tc.tab == 'tab2'
    assert tc._tab.currentIndex() == 1

    with pytest.raises(ValueError) as exc:
        tc.tab = 'tab3'
    assert exc.value.args[0] == "Cannot find value 'tab3' in tabs"


def test_connect_bool_button():

    class Test(object):
        a = CallbackProperty()

    t = Test()

    box = QCheckBox()
    connect_bool_button(t, 'a', box)

    box.setChecked(True)
    assert t.a

    box.setChecked(False)
    assert not t.a

    t.a = True
    assert box.isChecked()

    t.a = False
    assert not box.isChecked()


def test_connect_current_combo():

    class Test(object):
        a = CallbackProperty()

    t = Test()

    combo = QComboBox()
    combo.addItem('a', 'a')
    combo.addItem('b', 'b')

    connect_current_combo(t, 'a', combo)

    combo.setCurrentIndex(1)
    assert t.a == 'b'

    combo.setCurrentIndex(0)
    assert t.a == 'a'

    t.a = 'b'
    assert combo.currentIndex() == 1

    t.a = 'a'
    assert combo.currentIndex() == 0

    # TODO: should the following not return an error?
    t.a = 'c'
    assert combo.currentIndex() == 0


def test_connect_float_edit():

    class Test(object):
        a = CallbackProperty()

    t = Test()

    line = QLineEdit()
    dum = QWidget()

    connect_float_edit(t, 'a', line)

    event = QKeyEvent(QtCore.QEvent.KeyPress,
                      QtCore.Qt.Key_Return,
                      QtCore.Qt.NoModifier)

    app = get_qapp()

    line.setText('1.0')
    app.sendEvent(line, event)
    assert t.a == 1.0

    line.setText('4.0')
    app.sendEvent(line, event)
    assert t.a == 4.0

    t.a = 3.0
    assert line.text() == '3'


def test_connect_int_spin():

    class Test(object):
        a = CallbackProperty()

    t = Test()

    slider = QSlider()

    connect_int_spin(t, 'a', slider)

    slider.setValue(4)
    assert t.a == 4

    t.a = 3.0
    assert slider.value() == 3.0
