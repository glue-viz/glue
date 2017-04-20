# The functions in this module are used to connect callback properties to Qt
# widgets.

from __future__ import absolute_import, division, print_function

import math
from functools import partial

from .. import add_callback

__all__ = ['connect_checkable_button', 'connect_text', 'connect_combo_data',
           'connect_combo_text', 'connect_float_text', 'connect_value']


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
            setattr(instance, prop, widget.itemData(idx))

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
            return fmt.format(x)

    def update_prop():
        val = widget.text()
        try:
            setattr(instance, prop, float(val))
        except ValueError:
            setattr(instance, prop, 0)

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
    for idx in range(widget.count()):
        if widget.itemData(idx) == value:
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
