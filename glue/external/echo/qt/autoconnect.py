from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets

from .connect import (connect_checkable_button,
                      connect_value,
                      connect_combo_data,
                      connect_combo_text,
                      connect_float_text,
                      connect_text,
                      connect_button,
                      connect_combo_selection)

__all__ = ['autoconnect_callbacks_to_qt']

HANDLERS = {}
HANDLERS['value'] = connect_value
HANDLERS['valuetext'] = connect_float_text
HANDLERS['bool'] = connect_checkable_button
HANDLERS['text'] = connect_text
HANDLERS['combodata'] = connect_combo_data
HANDLERS['combotext'] = connect_combo_text
HANDLERS['button'] = connect_button
HANDLERS['combosel'] = connect_combo_selection


def autoconnect_callbacks_to_qt(instance, widget, connect_kwargs={}):
    """
    Given a class instance with callback properties and a Qt widget/window,
    connect callback properties to Qt widgets automatically.

    The matching is done based on the objectName of the Qt widgets. Qt widgets
    that need to be connected should be named using the syntax ``type_name``
    where ``type`` describes the kind of matching to be done, and ``name``
    matches the name of a callback property. By default, the types can be:

    * ``value``: the callback property is linked to a Qt widget that has
      ``value`` and ``setValue`` methods. Note that for this type, two
      additional keyword arguments can be specified using ``connect_kwargs``
      (see below): these are ``value_range``, which is used for cases where
      the Qt widget is e.g. a slider which has a range of values, and you want
      to map this range of values onto a different range for the callback
      property, and the second is ``log``, which can be set to `True` if this
      mapping should be done in log space.

    * ``valuetext``: the callback property is linked to a Qt widget that has
      ``text`` and ``setText`` methods, and the text is set to a string
      representation of the value. Note that for this type, an additional
      argument ``fmt`` can be provided, which gives either the format to use
      using the ``{}`` syntax, or should be a function that takes a value
      and returns a string. Optionally, if the Qt widget supports
      the ``editingFinished`` signal, this signal is connected to the callback
      property too.

    * ``bool``: the callback property is linked to a Qt widget that has
      ``isChecked`` and ``setChecked`` methods, such as a checkable button.

    * ``text``: the callback property is linked to a Qt widget that has
      ``text`` and ``setText`` methods. Optionally, if the Qt widget supports
      the ``editingFinished`` signal, this signal is connected to the callback
      property too.

    * ``combodata``: the callback property is linked to a QComboBox based on
      the ``userData`` of the entries in the combo box.

    * ``combotext``: the callback property is linked to a QComboBox based on
      the label of the entries in the combo box.

    Applications can also define additional mappings between type and
    auto-linking. To do this, simply add a new entry to the ``HANDLERS`` object::

        >>> echo.qt.autoconnect import HANDLERS
        >>> HANDLERS['color'] = connect_color

    The handler function (``connect_color`` in the example above) should take
    the following arguments: the instance the callback property is attached to,
    the name of the callback property, the Qt widget, and optionally some
    keyword arguments.

    When calling ``autoconnect_callbacks_to_qt``, you can specify
    ``connect_kwargs``, where each key should be a valid callback property name,
    and which gives any additional keyword arguments that can be taken by the
    connect functions, as described above. These include for example
    ``value_range``, ``log``, and ``fmt``.

    This function is especially useful when defining ui files, since widget
    objectNames can be easily set during the editing process.
    """

    if not hasattr(widget, 'children'):
        return

    for original_name in dir(widget):
        if original_name.startswith('_') or '_' not in original_name:
            continue
        # FIXME: this is a temorary workaround to allow multiple widgets to be
        # connected to a state attribute.
        if original_name.endswith('_'):
            full_name = original_name[:-1]
        else:
            full_name = original_name
        if '_' in full_name:
            wtype, wname = full_name.split('_', 1)
            if full_name in connect_kwargs:
                kwargs = connect_kwargs[full_name]
            elif wname in connect_kwargs:
                kwargs = connect_kwargs[wname]
            else:
                kwargs = {}
            if hasattr(instance, wname):
                if wtype in HANDLERS:
                    child = getattr(widget, original_name)
                    HANDLERS[wtype](instance, wname, child, **kwargs)
