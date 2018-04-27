from __future__ import absolute_import, division, print_function

import pytest

from qtpy import QtWidgets

from glue.external.echo.qt.connect import UserDataWrapper

from ..widget_properties import (CurrentComboDataProperty,
                                 CurrentComboTextProperty,
                                 CurrentTabProperty,
                                 TextProperty,
                                 ButtonProperty,
                                 FloatLineProperty,
                                 ValueProperty)


def test_combo_data():

    class TestClass(object):

        co = CurrentComboDataProperty('_combo')

        def __init__(self):
            self._combo = QtWidgets.QComboBox()
            self._combo.addItem('a', UserDataWrapper('a'))
            self._combo.addItem('b', UserDataWrapper('b'))

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
            self._combo = QtWidgets.QComboBox()
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
    assert tc.co is None
    assert tc._combo.currentIndex() == -1


def test_text():

    class TestClass(object):
        lab = TextProperty('_label')

        def __init__(self):
            self._label = QtWidgets.QLabel()

    tc = TestClass()
    tc.lab = 'hello'
    assert tc.lab == 'hello'
    assert tc._label.text() == 'hello'


def test_button():

    class TestClass(object):
        but = ButtonProperty('_button')

        def __init__(self):
            self._button = QtWidgets.QCheckBox()

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
            self._float = QtWidgets.QLineEdit()

    tc = TestClass()

    tc.flt = 1.0
    assert float(tc._float.text()) == 1.0

    tc._float.setText('10')
    assert tc.flt == 10.0

    tc._float.setText('')
    assert tc.flt == 0.0


def test_value():

    class TestClass(object):
        val1 = ValueProperty('_slider')
        val2 = ValueProperty('_slider', value_range=(0, 10))
        val3 = ValueProperty('_slider', value_range=(0.01, 100), log=True)

        def __init__(self):
            self._slider = QtWidgets.QSlider()
            self._slider.setMinimum(0)
            self._slider.setMaximum(100)

    tc = TestClass()

    tc.val1 = 2.0
    assert tc.val1 == 2.0
    assert tc._slider.value() == 2.0

    tc.val2 = 3.2
    assert tc.val2 == 3.2
    assert tc._slider.value() == 32

    tc.val3 = 10
    assert tc.val3 == 10
    assert tc._slider.value() == 75


def test_tab():

    class TestClass(object):
        tab = CurrentTabProperty('_tab')

        def __init__(self):
            self._tab = QtWidgets.QTabWidget()
            self._tab.addTab(QtWidgets.QWidget(), 'tab1')
            self._tab.addTab(QtWidgets.QWidget(), 'tab2')

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
