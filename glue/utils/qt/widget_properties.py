"""
The classes in this module provide a property-like interface
to widget instance variables in a class. These properties translate
essential pieces of widget state into more convenient python objects
(for example, the check state of a button to a bool).

Example Use::

    class Foo(object):
        bar = ButtonProperty('_button')

        def __init__(self):
            self._button = QtWidgets.QCheckBox()

    f = Foo()
    f.bar = True  # equivalent to f._button.setChecked(True)
    assert f.bar == True
"""

from __future__ import absolute_import, division, print_function

import math
from functools import partial

from qtpy import QtGui
from glue.logger import logger
from glue.external.six.moves import reduce
from glue.external.echo import add_callback
from glue.utils.array import pretty_number

__all__ = ['WidgetProperty', 'CurrentComboDataProperty',
           'CurrentComboTextProperty', 'CurrentTabProperty', 'TextProperty',
           'ButtonProperty', 'FloatLineProperty', 'ValueProperty',
           'connect_bool_button', 'connect_current_combo',
           'connect_float_edit', 'connect_int_spin']


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
        # Under certain circumstances, PyQt will try and access these properties
        # while loading the ui file, so we have to be robust to failures.
        # However, we print out a warning if things fail.
        try:
            widget = reduce(getattr, [instance] + self._att)
            return self.getter(widget)
        except Exception:
            logger.info("An error occured when accessing attribute {0} of {1}. Returning None.".format('.'.join(self._att), instance))
            return None

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
        if widget.currentIndex() == -1:
            return None
        else:
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
            if value is None:
                idx = -1
            else:
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
        if widget.currentIndex() == -1:
            return None
        else:
            return widget.itemText(widget.currentIndex())

    def setter(self, widget, value):
        """
        Update the currently selected item to the one which stores value in
        its itemData
        """
        idx = widget.findText(value)
        if idx == -1:
            if value is not None:
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
        if hasattr(widget, 'editingFinished'):
            widget.editingFinished.emit()


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
    value_range : tuple, optional
        If set, the slider values are mapped to this range.
    log : bool, optional
        If `True`, the mapping is assumed to be logarithmic instead of
        linear.
    """

    def __init__(self, att, docstring='',value_range=None, log=False):
        super(ValueProperty, self).__init__(att, docstring=docstring)

        if log:
            if value_range is None:
                raise ValueError("log option can only be set if value_range is given")
            else:
                value_range = math.log10(value_range[0]), math.log10(value_range[1])

        self.log = log
        self.value_range = value_range

    def getter(self, widget):
        val = widget.value()
        if self.value_range is not None:
            imin, imax = widget.minimum(), widget.maximum()
            vmin, vmax = self.value_range
            val = (val - imin) / (imax - imin) * (vmax - vmin) + vmin
        if self.log:
            val = 10 ** val
        return val

    def setter(self, widget, val):
        if self.log:
            val = math.log10(val)
        if self.value_range is not None:
            imin, imax = widget.minimum(), widget.maximum()
            vmin, vmax = self.value_range
            val = (val - vmin) / (vmax - vmin) * (imax - imin) + imin
        widget.setValue(val)


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

    def update_widget(value):
        try:
            idx = _find_combo_data(widget, value)
        except ValueError:
            if value is None:
                idx = -1
            else:
                raise
        widget.setCurrentIndex(idx)

    def update_prop(idx):
        if idx == -1:
            setattr(client, prop, None)
        else:
            setattr(client, prop, widget.itemData(idx))

    add_callback(client, prop, update_widget)
    widget.currentIndexChanged.connect(update_prop)
    update_widget(getattr(client, prop))


def connect_float_edit(client, prop, widget):
    """
    Connect widget.setText and client.prop
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


def connect_value(client, prop, widget, value_range=None, log=False):
    """
    Connect client.prop to widget.valueChanged

    client.prop should be a callback property

    If ``value_range`` is set, the slider values are mapped to that range. If
    ``log`` is set, the mapping is assumed to be logarithmic instead of linear.
    """

    if log:
        if value_range is None:
            raise ValueError("log option can only be set if value_range is given")
        else:
            value_range = math.log10(value_range[0]), math.log10(value_range[1])

    def update_prop():
        val = widget.value()
        if value_range is not None:
            imin, imax = widget.minimum(), widget.maximum()
            val = (val - imin) / (imax - imin) * (value_range[1] - value_range[0]) + value_range[0]
        if log:
            val = 10 ** val
        setattr(client, prop, val)

    def update_widget(val):
        if val is None:
            widget.setValue(0)
            return
        if log:
            val = math.log10(val)
        if value_range is not None:
            imin, imax = widget.minimum(), widget.maximum()
            val = (val - value_range[0]) / (value_range[1] - value_range[0]) * (imax - imin) + imin
        widget.setValue(val)

    add_callback(client, prop, update_widget)
    widget.valueChanged.connect(update_prop)
    update_widget(getattr(client, prop))


connect_int_spin = connect_value

def _find_combo_data(widget, value):
    """
    Returns the index in a combo box where itemData == value

    Raises a ValueError if data is not found
    """
    i = widget.findData(value)
    if i == -1:
        raise ValueError("{0} not found in combo box".format(value))
    else:
        return i
