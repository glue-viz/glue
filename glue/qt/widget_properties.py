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

from __future__ import absolute_import, division, print_function

from functools import partial

from .qtutil import pretty_number
from ..external.qt import QtGui
from ..external.six.moves import reduce
from ..core.callback_property import add_callback


class WidgetProperty(object):
    """
    Base class for widget properties

    Subclasses implement, at a minimum, the "get" and "set" methods,
    which translate between widget states and python variables

    Parameters
    ----------
    att : str
        The location, within a class instance, of the widget to wrap around.
        If the widget is nested inside another variable, normal '.' syntax
        can be used (e.g. 'sub_window.button')
    docstring : str, optional
        Optional short summary for the property. Used by sphinx. Should be 1
        sentence or less.
    """

    def __init__(self, att, docstring=''):
        self.__doc__ = docstring
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
        raise NotImplementedError()  # pragma: no cover

    def setter(self, widget, value):
        """ Set the state of a widget to a certain value"""
        raise NotImplementedError()  # pragma: no cover


class CurrentComboDataProperty(WidgetProperty):
    """
    Wrapper around the data in QComboBox.
    """

    def getter(self, widget):
        """
        Return the itemData stored in the currently-selected item
        """
        return widget.itemData(widget.currentIndex())

    def setter(self, widget, value):
        """
        Update the currently selected item to the one which stores value in
        its itemData
        """
        # Note, we don't use findData here because it doesn't work
        # well with non-str data
        try:
            idx = _find_combo_data(widget, value)
        except ValueError:
            raise ValueError("Cannot find data '{0}' in combo box".format(value))
        widget.setCurrentIndex(idx)

CurrentComboProperty = CurrentComboDataProperty


class CurrentComboTextProperty(WidgetProperty):
    """
    Wrapper around the text in QComboBox.
    """

    def getter(self, widget):
        """
        Return the itemData stored in the currently-selected item
        """
        return widget.itemText(widget.currentIndex())

    def setter(self, widget, value):
        """
        Update the currently selected item to the one which stores value in
        its itemData
        """
        idx = widget.findText(value)
        if idx == -1:
            raise ValueError("Cannot find text '{0}' in combo box".format(value))
        widget.setCurrentIndex(idx)


class CurrentTabProperty(WidgetProperty):
    """
    Wrapper around QTabWidget.
    """

    def getter(self, widget):
        """
        Return the itemData stored in the currently-selected item
        """
        return widget.tabText(widget.currentIndex())

    def setter(self, widget, value):
        """
        Update the currently selected item to the one which stores value in
        its itemData
        """
        for idx in range(widget.count()):
            if widget.tabText(idx) == value:
                break
        else:
            raise ValueError("Cannot find value '{0}' in tabs".format(value))

        widget.setCurrentIndex(idx)


class TextProperty(WidgetProperty):
    """
    Wrapper around the text() and setText() methods for QLabel etc
    """

    def getter(self, widget):
        return widget.text()

    def setter(self, widget, value):
        widget.setText(value)


class ButtonProperty(WidgetProperty):
    """
    Wrapper around the check state for QAbstractButton widgets
    """

    def getter(self, widget):
        return widget.isChecked()

    def setter(self, widget, value):
        widget.setChecked(value)


class FloatLineProperty(WidgetProperty):
    """
    Wrapper around the text state for QLineEdit widgets.

    Assumes that the text is a floating-point number
    """

    def getter(self, widget):
        try:
            return float(widget.text())
        except ValueError:
            return 0

    def setter(self, widget, value):
        widget.setText(pretty_number(value))
        widget.editingFinished.emit()


class ValueProperty(WidgetProperty):
    """
    Wrapper around widgets with value() and setValue()

    Parameters
    ----------
    att : str
        The location, within a class instance, of the widget to wrap around.
        If the widget is nested inside another variable, normal '.' syntax
        can be used (e.g. 'sub_window.button')
    docstring : str, optional
        Optional short summary for the property. Used by sphinx. Should be 1
        sentence or less.
    mapping : tuple, optional
        If specified, should be a tuple of two functions - the first to map
        from Qt widget values to Python values, and the second to map from
        Python values to Qt widget values. This can be used for to specify a
        non-linear mapping for sliders.
    """

    def __init__(self, att, docstring='', mapping=None):
        super(ValueProperty, self).__init__(att, docstring=docstring)
        self.mapping = mapping

    def getter(self, widget):
        if self.mapping is None:
            return widget.value()
        else:
            return self.mapping[0](widget.value())

    def setter(self, widget, value):
        if self.mapping is None:
            widget.setValue(value)
        else:
            widget.setValue(self.mapping[1](value))


def connect_bool_button(client, prop, widget):
    """ Connect widget.setChecked and client.prop

    client.prop should be a callback property
    """
    add_callback(client, prop, widget.setChecked)
    widget.toggled.connect(partial(setattr, client, prop))


def connect_current_combo(client, prop, widget):
    """
    Connect widget.currentIndexChanged and client.prop

    client.prop should be a callback property
    """

    def _push_combo(value):
        idx = widget.findData(value)
        if idx == -1:
            return
        widget.setCurrentIndex(idx)

    def _pull_combo(idx):
        setattr(client, prop, widget.itemData(idx))

    add_callback(client, prop, _push_combo)
    widget.currentIndexChanged.connect(_pull_combo)


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
        if val is None:
            val = 0.
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


def _find_combo_data(widget, value):
    """
    Returns the index in a combo box where itemData == value

    Raises a ValueError if data is not found
    """
    for i in range(widget.count()):
        if widget.itemData(i) == value:
            return i
    raise ValueError("%s not found in combo box" % value)