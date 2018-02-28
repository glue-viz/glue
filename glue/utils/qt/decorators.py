from __future__ import absolute_import, division, print_function

import sys
import traceback
from contextlib import contextmanager
from functools import wraps

__all__ = ['set_cursor', 'set_cursor_cm', 'messagebox_on_error',
           'die_on_error']


def set_cursor(shape):
    """Set the Qt cursor for the duration of a function call, and unset

    :param shape: Cursor shape to set.
    """
    def wrapper(func):
        @wraps(func)
        def result(*args, **kwargs):
            from glue.utils.qt import get_qapp  # Here to avoid circ import
            app = get_qapp()
            app.setOverrideCursor(shape)
            try:
                return func(*args, **kwargs)
            finally:
                app.restoreOverrideCursor()
        return result

    return wrapper


# TODO: Does this really belong in this module?
@contextmanager
def set_cursor_cm(shape):
    """Context manager equivalent for :func:`set_cursor`."""
    from glue.utils.qt import get_qapp
    app = get_qapp()
    app.setOverrideCursor(shape)
    try:
        yield
    finally:
        app.restoreOverrideCursor()


def messagebox_on_error(msg, sep='\n'):
    """Decorator that catches exceptions and displays an error message"""

    from qtpy import QtWidgets  # Must be here
    from qtpy.QtCore import Qt

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                m = "%s%s%s" % (msg, sep, str(e))
                detail = str(traceback.format_exc())
                qmb = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Error", m)
                qmb.setDetailedText(detail)
                qmb.resize(400, qmb.size().height())
                qmb.setTextInteractionFlags(Qt.TextSelectableByMouse)
                qmb.exec_()
        return wrapper

    return decorator


def die_on_error(msg):
    """Decorator that catches errors, displays a popup message, and quits"""

    from qtpy import QtWidgets  # Must be here

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Make sure application has been started
                from glue.utils.qt import get_qapp  # Here to avoid circ import
                get_qapp()

                m = "%s\n%s" % (msg, e)
                detail = str(traceback.format_exc())
                if len(m) > 500:
                    detail = "Full message:\n\n%s\n\n%s" % (m, detail)
                    m = m[:500] + '...'

                qmb = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Error", m)
                qmb.setDetailedText(detail)
                qmb.show()
                qmb.raise_()
                qmb.exec_()
                sys.exit(1)
        return wrapper
    return decorator
