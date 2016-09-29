from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets

from glue.utils.qt.widget_properties import (connect_bool_button,
                                             connect_value,
                                             connect_current_combo,
                                             connect_current_combo_text,
                                             connect_float_edit,
                                             connect_text)
from glue.utils.qt.colors import connect_color

__all__ = ['autoconnect_qt']


def autoconnect_qt(state, widget, connect_kwargs={}):

    # Could also optionally connect buttons to callback functions, e.g.
    # button_cancel to cancel method.

    if not hasattr(widget, 'children'):
        return

    for child in widget.findChildren(QtWidgets.QWidget):

        full_name = child.objectName()

        if '_' in full_name:

            wtype, wname = full_name.split('_', 1)

            kwargs = connect_kwargs.get(wname, {})

            if hasattr(state, wname):

                if wtype == 'value':
                    if isinstance(child, QtWidgets.QLineEdit):
                        connect_float_edit(state, wname, child, **kwargs)
                    else:
                        connect_value(state, wname, child, **kwargs)
                elif wtype == 'text':
                    connect_text(state, wname, child, **kwargs)
                elif wtype == 'bool':
                    connect_bool_button(state, wname, child, **kwargs)
                elif wtype == 'combo':
                    connect_current_combo(state, wname, child, **kwargs)
                elif wtype == 'combotext':  # TODO: find better name
                    connect_current_combo_text(state, wname, child, **kwargs)
                elif wtype == 'color':
                    connect_color(state, wname, child, **kwargs)
