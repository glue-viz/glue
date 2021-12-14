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

import math
from functools import reduce

from glue.logger import logger
from glue.utils.array import pretty_number

from echo.qt.connect import _find_combo_data, UserDataWrapper

# Backward-compatibility
from echo.qt import (connect_checkable_button as connect_bool_button,  # noqa
                                   connect_combo_data as connect_current_combo,  # noqa
                                   connect_combo_text as connect_current_combo_text,  # noqa
                                   connect_float_text as connect_float_edit,  # noqa
                                   connect_value, connect_text)  # noqa

connect_int_spin = connect_value

__all__ = ['WidgetProperty', 'CurrentComboDataProperty',
           'CurrentComboTextProperty', 'CurrentTabProperty', 'TextProperty',
           'ButtonProperty', 'FloatLineProperty', 'ValueProperty']


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
            data = widget.itemData(widget.currentIndex())
            if isinstance(data, UserDataWrapper):
                return data.data
            else:
                return data

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

    def __init__(self, att, docstring='', value_range=None, log=False):
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
        widget.setValue(int(val))
