import pytest

from qtpy import QtWidgets

from glue.external.echo.core import CallbackProperty
from glue.external.echo.selection import SelectionCallbackProperty, ChoiceSeparator
from glue.external.echo.qt.connect import connect_combo_selection


class Example(object):
    a = SelectionCallbackProperty(default_index=1)
    b = CallbackProperty()


def test_connect_combo_selection():

    t = Example()

    a_prop = getattr(type(t), 'a')
    a_prop.set_choices(t, [4, 3.5])
    a_prop.set_display_func(t, lambda x: 'value: {0}'.format(x))

    combo = QtWidgets.QComboBox()

    c = connect_combo_selection(t, 'a', combo)

    assert combo.itemText(0) == 'value: 4'
    assert combo.itemText(1) == 'value: 3.5'
    assert combo.itemData(0).data == 4
    assert combo.itemData(1).data == 3.5

    combo.setCurrentIndex(1)
    assert t.a == 3.5

    combo.setCurrentIndex(0)
    assert t.a == 4

    combo.setCurrentIndex(-1)
    assert t.a is None

    t.a = 3.5
    assert combo.currentIndex() == 1

    t.a = 4
    assert combo.currentIndex() == 0

    with pytest.raises(ValueError) as exc:
        t.a = 2
    assert exc.value.args[0] == 'value 2 is not in valid choices: [4, 3.5]'

    t.a = None
    assert combo.currentIndex() == -1

    # Changing choices should change Qt combo box. Let's first try with a case
    # in which there is a matching data value in the new combo box

    t.a = 3.5
    assert combo.currentIndex() == 1

    a_prop.set_choices(t, (4, 5, 3.5))
    assert combo.count() == 3

    assert t.a == 3.5
    assert combo.currentIndex() == 2

    assert combo.itemText(0) == 'value: 4'
    assert combo.itemText(1) == 'value: 5'
    assert combo.itemText(2) == 'value: 3.5'
    assert combo.itemData(0).data == 4
    assert combo.itemData(1).data == 5
    assert combo.itemData(2).data == 3.5

    # Now we change the choices so that there is no matching data - in this case
    # the index should change to that given by default_index

    a_prop.set_choices(t, (4, 5, 6))

    assert t.a == 5
    assert combo.currentIndex() == 1
    assert combo.count() == 3

    assert combo.itemText(0) == 'value: 4'
    assert combo.itemText(1) == 'value: 5'
    assert combo.itemText(2) == 'value: 6'
    assert combo.itemData(0).data == 4
    assert combo.itemData(1).data == 5
    assert combo.itemData(2).data == 6

    # Finally, if there are too few choices for the default_index to be valid,
    # pick the last item in the combo

    a_prop.set_choices(t, (9,))

    assert t.a == 9
    assert combo.currentIndex() == 0
    assert combo.count() == 1

    assert combo.itemText(0) == 'value: 9'
    assert combo.itemData(0).data == 9

    # Now just make sure that ChoiceSeparator works

    separator = ChoiceSeparator('header')
    a_prop.set_choices(t, (separator, 1, 2))

    assert combo.count() == 3
    assert combo.itemText(0) == 'header'
    assert combo.itemData(0).data is separator

    # And setting choices to an empty iterable shouldn't cause issues

    a_prop.set_choices(t, ())
    assert combo.count() == 0


def test_connect_combo_selection_invalid():

    t = Example()

    combo = QtWidgets.QComboBox()

    with pytest.raises(TypeError) as exc:
        connect_combo_selection(t, 'b', combo)
    assert exc.value.args[0] == 'connect_combo_selection requires a SelectionCallbackProperty'
