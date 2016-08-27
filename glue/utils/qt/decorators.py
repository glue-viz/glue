from __future__ import absolute_import, division, print_function

import traceback
from contextlib import contextmanager
from functools import wraps

from glue.utils.qt import QMessageBoxPatched as QMessageBox

__all__ = ['set_cursor', 'set_cursor_cm', 'messagebox_on_error']


def set_cursor(shape):
    """Set the Qt cursor for the duration of a function call, and unset

    :param shape: Cursor shape to set.
    """
    def wrapper(func):
        @wraps(func)
        def result(*args, **kwargs):
            from glue.utils.qt import get_qapp
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


def messagebox_on_error(msg):
    """Decorator that catches exceptions and displays an error message"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                m = "%s\n%s" % (msg, e.args[0])
                detail = str(traceback.format_exc())
                qmb = QMessageBox(QMessageBox.Critical, "Error", m)
                qmb.setDetailedText(detail)
                qmb.resize(400, qmb.size().height())
                qmb.exec_()
        return wrapper

    return decorator
