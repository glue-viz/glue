# The functions in this module are used to connect callback properties to Qt
# widgets.

from __future__ import absolute_import, division, print_function

import math
from functools import partial

from qtpy import QtGui
from qtpy.QtCore import Qt

import numpy as np

from ..core import add_callback
from ..selection import SelectionCallbackProperty, ChoiceSeparator

__all__ = ['connect_checkable_button', 'connect_text', 'connect_combo_data',
           'connect_combo_text', 'connect_float_text', 'connect_value',
           'connect_combo_selection']


class UserDataWrapper(object):
    def __init__(self, data):
        self.data = data


def connect_checkable_button(instance, prop, widget):
    """
    Connect a boolean callback property with a Qt button widget.

    Parameters
    ----------
    instance : object
        The class instance that the callback property is attached to
    prop : str
        The name of the callback property
    widget : QtWidget
        The Qt widget to connect. This should implement the ``setChecked``
        method and the ``toggled`` signal.
    """
    add_callback(instance, prop, widget.setChecked)
    widget.toggled.connect(partial(setattr, instance, prop))
    widget.setChecked(getattr(instance, prop) or False)


def connect_text(instance, prop, widget):
    """
    Connect a string callback property with a Qt widget containing text.

    Parameters
    ----------
    instance : object
        The class instance that the callback property is attached to
    prop : str
        The name of the callback property
    widget : QtWidget
        The Qt widget to connect. This should implement the ``setText`` and
        ``text`` methods as well optionally the ``editingFinished`` signal.
    """

    def update_prop():
        val = widget.text()
        setattr(instance, prop, val)

    def update_widget(val):
        if hasattr(widget, 'editingFinished'):
            widget.blockSignals(True)
            widget.setText(val)
            widget.blockSignals(False)
            widget.editingFinished.emit()
        else:
            widget.setText(val)

    add_callback(instance, prop, update_widget)

    try:
        widget.editingFinished.connect(update_prop)
    except AttributeError:
        pass

    update_widget(getattr(instance, prop))


def connect_combo_data(instance, prop, widget):
    """
    Connect a callback property with a QComboBox widget based on the userData.

    Parameters
    ----------
    instance : object
        The class instance that the callback property is attached to
    prop : str
        The name of the callback property
    widget : QComboBox
        The combo box to connect.

    See Also
    --------
    connect_combo_text: connect a callback property with a QComboBox widget based on the text.
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
            setattr(instance, prop, None)
        else:
            data_wrapper = widget.itemData(idx)
            if data_wrapper is None:
                setattr(instance, prop, None)
            else:
                setattr(instance, prop, data_wrapper.data)

    add_callback(instance, prop, update_widget)
    widget.currentIndexChanged.connect(update_prop)

    update_widget(getattr(instance, prop))


def connect_combo_text(instance, prop, widget):
    """
    Connect a callback property with a QComboBox widget based on the text.

    Parameters
    ----------
    instance : object
        The class instance that the callback property is attached to
    prop : str
        The name of the callback property
    widget : QComboBox
        The combo box to connect.

    See Also
    --------
    connect_combo_data: connect a callback property with a QComboBox widget based on the userData.
    """

    def update_widget(value):
        try:
            idx = _find_combo_text(widget, value)
        except ValueError:
            if value is None:
                idx = -1
            else:
                raise
        widget.setCurrentIndex(idx)

    def update_prop(idx):
        if idx == -1:
            setattr(instance, prop, None)
        else:
            setattr(instance, prop, widget.itemText(idx))

    add_callback(instance, prop, update_widget)
    widget.currentIndexChanged.connect(update_prop)

    update_widget(getattr(instance, prop))


def connect_float_text(instance, prop, widget, fmt="{:g}"):
    """
    Connect a numerical callback property with a Qt widget containing text.

    Parameters
    ----------
    instance : object
        The class instance that the callback property is attached to
    prop : str
        The name of the callback property
    widget : QtWidget
        The Qt widget to connect. This should implement the ``setText`` and
        ``text`` methods as well optionally the ``editingFinished`` signal.
    fmt : str or func
        This should be either a format string (in the ``{}`` notation), or a
        function that takes a number and returns a string.
    """

    if callable(fmt):
        format_func = fmt
    else:
        def format_func(x):
            try:
                return fmt.format(x)
            except ValueError:
                return str(x)

    def update_prop():
        val = widget.text()
        try:
            val = float(val)
        except ValueError:
            try:
                val = np.datetime64(val)
            except Exception:
                val = 0
        setattr(instance, prop, val)

    def update_widget(val):
        if val is None:
            val = 0.
        widget.setText(format_func(val))

    add_callback(instance, prop, update_widget)

    try:
        widget.editingFinished.connect(update_prop)
    except AttributeError:
        pass

    update_widget(getattr(instance, prop))


def connect_value(instance, prop, widget, value_range=None, log=False):
    """
    Connect a numerical callback property with a Qt widget representing a value.

    Parameters
    ----------
    instance : object
        The class instance that the callback property is attached to
    prop : str
        The name of the callback property
    widget : QtWidget
        The Qt widget to connect. This should implement the ``setText`` and
        ``text`` methods as well optionally the ``editingFinished`` signal.
    value_range : iterable, optional
        A pair of two values representing the true range of values (since
        Qt widgets such as sliders can only have values in certain ranges).
    log : bool, optional
        Whether the Qt widget value should be mapped to the log of the callback
        property.
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
        setattr(instance, prop, val)

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

    add_callback(instance, prop, update_widget)
    widget.valueChanged.connect(update_prop)

    update_widget(getattr(instance, prop))


def connect_button(instance, prop, widget):
    """
    Connect a button with a callback method

    Parameters
    ----------
    instance : object
        The class instance that the callback method is attached to
    prop : str
        The name of the callback method
    widget : QtWidget
        The Qt widget to connect. This should implement the ``clicked`` method
    """
    widget.clicked.connect(getattr(instance, prop))


def _find_combo_data(widget, value):
    """
    Returns the index in a combo box where itemData == value

    Raises a ValueError if data is not found
    """
    # Here we check that the result is True, because some classes may overload
    # == and return other kinds of objects whether true or false.
    for idx in range(widget.count()):
        if widget.itemData(idx) is not None:
            if isinstance(widget.itemData(idx), UserDataWrapper):
                if widget.itemData(idx).data is value or (widget.itemData(idx).data == value) is True:
                    return idx
            else:
                if widget.itemData(idx) is value or (widget.itemData(idx) == value) is True:
                    return idx
    else:
        raise ValueError("%s not found in combo box" % (value,))


def _find_combo_text(widget, value):
    """
    Returns the index in a combo box where text == value

    Raises a ValueError if data is not found
    """
    i = widget.findText(value)
    if i == -1:
        raise ValueError("%s not found in combo box" % value)
    else:
        return i


def connect_combo_selection(instance, prop, widget, display=str):

    if not isinstance(getattr(type(instance), prop), SelectionCallbackProperty):
        raise TypeError('connect_combo_selection requires a SelectionCallbackProperty')

    def update_widget(value):

        # Update choices in the combo box

        combo_data = [widget.itemData(idx) for idx in range(widget.count())]
        combo_text = [widget.itemText(idx) for idx in range(widget.count())]

        choices = getattr(type(instance), prop).get_choices(instance)
        choice_labels = getattr(type(instance), prop).get_choice_labels(instance)

        if combo_data == choices and combo_text == choice_labels:
            choices_updated = False
        else:

            widget.blockSignals(True)
            widget.clear()

            if len(choices) == 0:
                return

            combo_model = widget.model()

            for index, (label, choice) in enumerate(zip(choice_labels, choices)):

                widget.addItem(label, userData=UserDataWrapper(choice))

                # We interpret None data as being disabled rows (used for headers)
                if isinstance(choice, ChoiceSeparator):
                    item = combo_model.item(index)
                    palette = widget.palette()
                    item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
                    item.setData(palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))

            choices_updated = True

        # Update current selection
        try:
            idx = _find_combo_data(widget, value)
        except ValueError:
            if value is None:
                idx = -1
            else:
                raise

        if idx == widget.currentIndex() and not choices_updated:
            return

        widget.setCurrentIndex(idx)
        widget.blockSignals(False)
        widget.currentIndexChanged.emit(idx)

    def update_prop(idx):
        if idx == -1:
            setattr(instance, prop, None)
        else:
            data_wrapper = widget.itemData(idx)
            if data_wrapper is None:
                setattr(instance, prop, None)
            else:
                setattr(instance, prop, data_wrapper.data)

    add_callback(instance, prop, update_widget)
    widget.currentIndexChanged.connect(update_prop)

    update_widget(getattr(instance, prop))
