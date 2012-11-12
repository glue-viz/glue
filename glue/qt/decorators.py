from functools import wraps
import traceback
from PyQt4.QtGui import QMessageBox


def set_cursor(shape):
    """Set the Qt cursor for the duration of a function call, and unset

    :param shape: Cursor shape to set.
    """
    def wrapper(func):
        @wraps(func)
        def result(*args, **kwargs):
            from . import get_qapp
            app = get_qapp()
            app.setOverrideCursor(shape)
            try:
                return func(*args, **kwargs)
            finally:
                app.restoreOverrideCursor()
        return result

    return wrapper


def messagebox_on_error(msg):
    """Decorator that catches exceptions and displays an error message"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                m = "%s\n%s" % (msg, e.message)
                detail = str(traceback.format_exc())
                qmb = QMessageBox(QMessageBox.Critical, "Error", m)
                qmb.setDetailedText(detail)
                qmb.resize(400, qmb.size().height())
                qmb.exec_()
        return wrapper

    return decorator
