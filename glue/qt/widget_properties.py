"""
The classes in this module provide a property-like interface
to widget instance variables in a class. These properties translate
essential pieces of widget state into more convenient python objects
(for example, the check state of a button to a bool).

Example Use::

    class Foo(object):
        bar = ButtonProperty('_button')

        def __init__(self):
            self._button = QtGui.QCheckBox()

    f = Foo()
    f.bar = True  # equivalent to f._button.setChecked(True)
    assert f.bar == True
"""
from functools import partial

from .qtutil import pretty_number
from ..external.qt import QtGui
from ..core.callback_property import add_callback


class WidgetProperty(object):
    """ Base class for widget properties

    Subclasses implement, at a minimum, the "get" and "set" methods,
    which translate between widget states and python variables
    """
    def __init__(self, att):
        """
        :param att: The location, within a class instance, of the widget
        to wrap around. If the widget is nested inside another variable,
        normal '.' syntax can be used (e.g. 'sub_window.button')

        :type att: str"""
        self._att = att.split('.')

    def __get__(self, instance, type=None):
        widget = reduce(getattr, [instance] + self._att)
        return self.getter(widget)

    def __set__(self, instance, value):
        widget = reduce(getattr, [instance] + self._att)
        self.setter(widget, value)

    def getter(self, widget):
        """ Return the state of a widget. Depends on type of widget,
        and must be overridden"""
        raise NotImplementedError()

    def setter(self, widget, value):
        """ Set the state of a widget to a certain value"""
        raise NotImplementedError()


class CurrentComboProperty(WidgetProperty):
    """Wrapper around ComboBoxes"""
    def getter(self, widget):
        """ Return the itemData stored in the currently-selected item """
        return widget.itemData(widget.currentIndex())

    def setter(self, widget, value):
        """ Update the currently selected item to the one which stores value in
        its itemData
        """
        for i in range(widget.count()):
            if widget.itemData(i) is value:
                widget.setCurrentIndex(i)
                return
        else:
            raise ValueError("%s not found in combo box" % value)


class ButtonProperty(WidgetProperty):
    """Wrapper around the check state for QAbstractButton widgets"""
    def getter(self, widget):
        return widget.isChecked()

    def setter(self, widget, value):
        widget.setChecked(value)


class FloatLineProperty(WidgetProperty):
    """Wrapper around the text state for QLineEdit widgets.

    Assumes that the text is a floating point number
    """
    def getter(self, widget):
        try:
            return float(widget.text())
        except ValueError:
            return 0

    def setter(self, widget, value):
        widget.setText(pretty_number(value))
        widget.editingFinished.emit()


class SpinnerProperty(WidgetProperty):
    """Wrapper around intspin boxes"""
    def getter(self, widget):
        return widget.value()

    def setter(self, widget, value):
        widget.setValue(value)



def connect_bool_button(client, prop, widget):
    """ Connect widget.setChecked and client.prop

    client.prop should be a callback property
    """
    add_callback(client, prop, widget.setChecked)
    widget.toggled.connect(partial(setattr, client, prop))


def connect_float_edit(client, prop, widget):
    """ Connect widget.setText and client.prop
    Also pretty-print the number

    client.prop should be a callback property
    """
    v = QtGui.QDoubleValidator(None)
    v.setDecimals(4)
    widget.setValidator(v)

    def update_prop():
        val = widget.text()
        try:
            setattr(client, prop, float(val))
        except ValueError:
            setattr(client, prop, 0)

    def update_widget(val):
        widget.setText(pretty_number(val))

    add_callback(client, prop, update_widget)
    widget.editingFinished.connect(update_prop)
    update_widget(getattr(client, prop))


def connect_int_spin(client, prop, widget):
    """
    Connect client.prop to widget.valueChanged

    client.prop should be a callback property
    """
    add_callback(client, prop, widget.setValue)
    widget.valueChanged.connect(partial(setattr, client, prop))
