from __future__ import absolute_import, division, print_function

import pytest

from glue.external.echo import CallbackProperty
from glue.external.qt import QtGui

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


def test_combo_data():

    class TestClass(object):

        co = CurrentComboDataProperty('_combo')

        def __init__(self):
            self._combo = QtGui.QComboBox()
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
            self._combo = QtGui.QComboBox()
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

    tc.co = None
    assert tc.co == None
    assert tc._combo.currentIndex() == -1


def test_text():

    class TestClass(object):
        lab = TextProperty('_label')

        def __init__(self):
            self._label = QtGui.QLabel()

    tc = TestClass()
    tc.lab = 'hello'
    assert tc.lab == 'hello'
    assert tc._label.text() == 'hello'


def test_button():

    class TestClass(object):
        but = ButtonProperty('_button')

        def __init__(self):
            self._button = QtGui.QCheckBox()

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
            self._float = QtGui.QLineEdit()

    tc = TestClass()

    tc.flt = 1.0
    assert float(tc._float.text()) == 1.0

    tc._float.setText('10')
    assert tc.flt == 10.0

    tc._float.setText('')
    assert tc.flt == 0.0


def test_value():

    class TestClass(object):
        val1 = ValueProperty('_slider1')
        val2 = ValueProperty('_slider2', value_range=(0, 10))
        val3 = ValueProperty('_slider3', value_range=(0.01, 100), log=True)

        def __init__(self):
            self._slider1 = QtGui.QSlider()
            self._slider2 = QtGui.QSlider()
            self._slider2.setMinimum(0)
            self._slider2.setMaximum(100)
            self._slider3 = QtGui.QSlider()
            self._slider3.setMinimum(0)
            self._slider3.setMaximum(100)

    tc = TestClass()

    tc.val1 = 2.0
    assert tc.val1 == 2.0
    assert tc._slider1.value() == 2.0

    tc.val2 = 3.2
    assert tc.val2 == 3.2
    assert tc._slider2.value() == 32

    tc.val3 = 10
    assert tc.val3 == 10
    assert tc._slider3.value() == 75


def test_tab():

    class TestClass(object):
        tab = CurrentTabProperty('_tab')

        def __init__(self):
            self._tab = QtGui.QTabWidget()
            self._tab.addTab(QtGui.QWidget(), 'tab1')
            self._tab.addTab(QtGui.QWidget(), 'tab2')

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

    box = QtGui.QCheckBox()
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

    combo = QtGui.QComboBox()
    combo.addItem('a', 'a')
    combo.addItem('b', 'b')

    connect_current_combo(t, 'a', combo)

    combo.setCurrentIndex(1)
    assert t.a == 'b'

    combo.setCurrentIndex(0)
    assert t.a == 'a'

    combo.setCurrentIndex(-1)
    assert t.a is None

    t.a = 'b'
    assert combo.currentIndex() == 1

    t.a = 'a'
    assert combo.currentIndex() == 0

    # TODO: should the following not return an error?
    with pytest.raises(ValueError) as exc:
        t.a = 'c'
    assert exc.value.args[0] == 'c not found in combo box'

    t.a = None
    assert combo.currentIndex() == -1


def test_connect_float_edit():

    class Test(object):
        a = CallbackProperty()

    t = Test()

    line = QtGui.QLineEdit()

    connect_float_edit(t, 'a', line)

    line.setText('1.0')
    line.editingFinished.emit()
    assert t.a == 1.0

    line.setText('4.0')
    line.editingFinished.emit()
    assert t.a == 4.0

    t.a = 3.0
    assert line.text() == '3'


def test_connect_int_spin():

    class Test(object):
        a = CallbackProperty()

    t = Test()

    slider = QtGui.QSlider()

    connect_int_spin(t, 'a', slider)

    slider.setValue(4)
    assert t.a == 4

    t.a = 3.0
    assert slider.value() == 3.0
