# The functions in this module are used to connect callback properties to Qt
# widgets.

from __future__ import absolute_import, division, print_function

import math
from functools import partial

from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt

import numpy as np

from ..core import add_callback, remove_callback
from ..selection import SelectionCallbackProperty, ChoiceSeparator

__all__ = ['connect_checkable_button', 'connect_text', 'connect_combo_data',
           'connect_combo_text', 'connect_float_text', 'connect_value',
           'connect_combo_selection', 'connect_list_selection',
           'BaseConnection']


class UserDataWrapper(object):
    def __init__(self, data):
        self.data = data


class BaseConnection(object):

    def __init__(self, instance, prop, widget):

        self._instance = instance
        self._prop = prop
        self._widget = widget


class connect_checkable_button(BaseConnection):
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

    def __init__(self, instance, prop, widget):
        super(connect_checkable_button, self).__init__(instance, prop, widget)
        self.connect()

    def update_widget(self, value):
        self._widget.setChecked(value)

    def update_prop(self, value):
        setattr(self._instance, self._prop, value)

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        self._widget.toggled.connect(self.update_prop)
        self._widget.setChecked(getattr(self._instance, self._prop) or False)

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        self._widget.toggled.disconnect(self.update_prop)


class connect_text(BaseConnection):
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

    def __init__(self, instance, prop, widget):
        super(connect_text, self).__init__(instance, prop, widget)
        self.connect()

    def update_prop(self):
        value = self._widget.text()
        setattr(self._instance, self._prop, value)

    def update_widget(self, value):
        if hasattr(self._widget, 'editingFinished'):
            self._widget.blockSignals(True)
            self._widget.setText(value)
            self._widget.blockSignals(False)
            self._widget.editingFinished.emit()
        else:
            self._widget.setText(value)

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        try:
            self._widget.editingFinished.connect(self.update_prop)
        except AttributeError:
            pass
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        try:
            self._widget.editingFinished.disconnect(self.update_prop)
        except AttributeError:
            pass


class connect_combo_data(BaseConnection):
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

    def __init__(self, instance, prop, widget):
        super(connect_combo_data, self).__init__(instance, prop, widget)
        self.connect()

    def update_widget(self, value):
        try:
            idx = _find_combo_data(self._widget, value)
        except ValueError:
            if value is None:
                idx = -1
            else:
                raise
        self._widget.setCurrentIndex(idx)

    def update_prop(self, idx):
        if idx == -1:
            setattr(self._instance, self._prop, None)
        else:
            data_wrapper = self._widget.itemData(idx)
            if data_wrapper is None:
                setattr(self._instance, self._prop, None)
            else:
                setattr(self._instance, self._prop, data_wrapper.data)

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        self._widget.currentIndexChanged.connect(self.update_prop)
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        self._widget.currentIndexChanged.disconnect(self.update_prop)


class connect_combo_text(BaseConnection):
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

    def __init__(self, instance, prop, widget):
        super(connect_combo_text, self).__init__(instance, prop, widget)
        self.connect()

    def update_widget(self, value):
        try:
            idx = _find_combo_text(self._widget, value)
        except ValueError:
            if value is None:
                idx = -1
            else:
                raise
        self._widget.setCurrentIndex(idx)

    def update_prop(self, idx):
        if idx == -1:
            setattr(self._instance, self._prop, None)
        else:
            setattr(self._instance, self._prop, self._widget.itemText(idx))

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        self._widget.currentIndexChanged.connect(self.update_prop)
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        self._widget.currentIndexChanged.disconnect(self.update_prop)


class connect_float_text(BaseConnection):
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

    def __init__(self, instance, prop, widget, fmt="{:g}"):

        super(connect_float_text, self).__init__(instance, prop, widget)

        if callable(fmt):
            format_func = fmt
        else:
            def format_func(x):
                try:
                    return fmt.format(x)
                except ValueError:
                    return str(x)

        self._format_func = format_func

        self.connect()

    def update_prop(self):
        value = self._widget.text()
        try:
            value = float(value)
        except ValueError:
            try:
                value = np.datetime64(value)
            except Exception:
                value = 0
        setattr(self._instance, self._prop, value)

    def update_widget(self, value):
        if value is None:
            value = 0.
        self._widget.setText(self._format_func(value))

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        try:
            self._widget.editingFinished.connect(self.update_prop)
        except AttributeError:
            pass
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        try:
            self._widget.editingFinished.disconnect(self.update_prop)
        except AttributeError:
            pass


class connect_value(BaseConnection):
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

    def __init__(self, instance, prop, widget, value_range=None, log=False):

        super(connect_value, self).__init__(instance, prop, widget)

        if log:
            if value_range is None:
                raise ValueError("log option can only be set if value_range is given")
            else:
                self._value_range = math.log10(value_range[0]), math.log10(value_range[1])
        else:
            self._value_range = value_range
        self._log = log

        self.connect()

    def update_prop(self):
        value = self._widget.value()
        if self._value_range is not None:
            imin, imax = self._widget.minimum(), self._widget.maximum()
            value = (value - imin) / (imax - imin) * (self._value_range[1] - self._value_range[0]) + self._value_range[0]
        if self._log:
            value = 10 ** value
        setattr(self._instance, self._prop, value)

    def update_widget(self, value):
        if value is None:
            self._widget.setValue(0)
            return
        if self._log:
            value = math.log10(value)
        if self._value_range is not None:
            imin, imax = self._widget.minimum(), self._widget.maximum()
            value = (value - self._value_range[0]) / (self._value_range[1] - self._value_range[0]) * (imax - imin) + imin
        self._widget.setValue(value)

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        self._widget.valueChanged.connect(self.update_prop)
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        self._widget.valueChanged.disconnect(self.update_prop)


class connect_button(BaseConnection):
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

    def __init__(self, instance, prop, widget):
        super(connect_button, self).__init__(instance, prop, widget)
        self.connect()

    def connect(self):
        self._widget.clicked.connect(getattr(self._instance, self._prop))

    def disconnect(self):
        self._widget.clicked.disconnect(self.update_prop)


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


class connect_combo_selection(BaseConnection):

    def __init__(self, instance, prop, widget):

        if not isinstance(getattr(type(instance), prop), SelectionCallbackProperty):
            raise TypeError('connect_combo_selection requires a SelectionCallbackProperty')

        super(connect_combo_selection, self).__init__(instance, prop, widget)
        self.connect()

    def update_widget(self, value):

        # Update choices in the combo box

        combo_data = [self._widget.itemData(idx) for idx in range(self._widget.count())]
        combo_text = [self._widget.itemText(idx) for idx in range(self._widget.count())]

        choices = getattr(type(self._instance), self._prop).get_choices(self._instance)
        choice_labels = getattr(type(self._instance), self._prop).get_choice_labels(self._instance)

        if combo_data == choices and combo_text == choice_labels:
            choices_updated = False
        else:

            self._widget.blockSignals(True)
            self._widget.clear()

            if len(choices) == 0:
                return

            combo_model = self._widget.model()

            for index, (label, choice) in enumerate(zip(choice_labels, choices)):

                self._widget.addItem(label, userData=UserDataWrapper(choice))

                # We interpret None data as being disabled rows (used for headers)
                if isinstance(choice, ChoiceSeparator):
                    item = combo_model.item(index)
                    palette = self._widget.palette()
                    item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
                    item.setData(palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))

            choices_updated = True

        # Update current selection
        try:
            idx = _find_combo_data(self._widget, value)
        except ValueError:
            if value is None:
                idx = -1
            else:
                raise

        if idx == self._widget.currentIndex() and not choices_updated:
            return

        self._widget.setCurrentIndex(idx)
        self._widget.blockSignals(False)
        self._widget.currentIndexChanged.emit(idx)

    def update_prop(self, idx):
        if idx == -1:
            setattr(self._instance, self._prop, None)
        else:
            data_wrapper = self._widget.itemData(idx)
            if data_wrapper is None:
                setattr(self._instance, self._prop, None)
            else:
                setattr(self._instance, self._prop, data_wrapper.data)

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        self._widget.currentIndexChanged.connect(self.update_prop)
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        self._widget.currentIndexChanged.disconnect(self.update_prop)


class connect_list_selection(BaseConnection):

    def __init__(self, instance, prop, widget):
        """
        Connect a SelectionCallbackProperty with a QListWidget that supports
        single-item selection.
        """

        if not isinstance(getattr(type(instance), prop), SelectionCallbackProperty):
            raise TypeError('connect_list_selection requires a SelectionCallbackProperty')

        super(connect_list_selection, self).__init__(instance, prop, widget)
        self.connect()

    def update_widget(self, value, force=False):

        items = [self._widget.item(idx) for idx in range(self._widget.count())]
        list_text = [item.text() for item in items]
        list_data = [item.data(Qt.UserRole) for item in items]
        list_data = [d.data if d is not None else d for d in list_data]

        choices = getattr(type(self._instance), self._prop).get_choices(self._instance)
        choice_labels = getattr(type(self._instance), self._prop).get_choice_labels(self._instance)

        for idx in range(len(choices)):
            if choices[idx] is value:
                break
        else:
            idx = -1

        self._widget.blockSignals(True)

        choices_match = list_data == choices and list_text == choice_labels

        if force or not choices_match:

            self._widget.clear()

            if len(choices) == 0:
                self._widget.blockSignals(False)
                return

            for index, (label, choice) in enumerate(zip(choice_labels, choices)):

                item = QtWidgets.QListWidgetItem(label)
                item.setData(Qt.UserRole, UserDataWrapper(choice))
                self._widget.addItem(item)

                # We interpret None data as being disabled rows (used for headers)
                if isinstance(choice, ChoiceSeparator):
                    palette = self._widget.palette()
                    item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
                    # item.setData(palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Text))

        if len(self._widget.selectedItems()) == 0:
            current_index = -1
        else:
            selected_item = self._widget.selectedItems()[0]
            for current_index, item in enumerate(items):
                if item is selected_item:
                    break
            else:
                current_index = -1

        if idx == current_index and choices_match:
            self._widget.blockSignals(False)
            return

        self._widget.setCurrentItem(self._widget.item(idx))
        self._widget.blockSignals(False)
        self._widget.itemSelectionChanged.emit()

    def update_prop(self):

        if len(self._widget.selectedItems()) == 0:
            setattr(self._instance, self._prop, None)
        else:
            data_wrapper = self._widget.selectedItems()[0].data(Qt.UserRole)
            if data_wrapper is None:
                setattr(self._instance, self._prop, None)
            else:
                setattr(self._instance, self._prop, data_wrapper.data)

    def connect(self):
        add_callback(self._instance, self._prop, self.update_widget)
        self._widget.itemSelectionChanged.connect(self.update_prop)
        self.update_widget(getattr(self._instance, self._prop))

    def disconnect(self):
        remove_callback(self._instance, self._prop, self.update_widget)
        self._widget.itemSelectionChanged.disconnect(self.update_prop)
