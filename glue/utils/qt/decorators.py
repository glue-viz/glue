import os
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


# TODO: We should be able to avoid defining these as classes and defining
# __call__ below, by using contextmanager.

class messagebox_on_error(object):

    def __init__(self, msg, sep='\n', exit=False):
        self.msg = msg
        self.sep = sep
        self.exit = exit

    def __call__(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # If we are in glue testing mode, just execute function
            if os.environ.get("GLUE_TESTING") == 'True':
                return f(*args, **kwargs)
            with self:
                return f(*args, **kwargs)
        return decorated

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, tb):

        if exc_type is None:
            return

        # Make sure application has been started
        from glue.utils.qt import get_qapp  # Here to avoid circular import
        get_qapp()

        m = "%s\n%s" % (self.msg, exc_val)
        detail = ''.join(traceback.format_exception(exc_type, exc_val, tb))
        if len(m) > 500:
            detail = "Full message:\n\n%s\n\n%s" % (m, detail)
            m = m[:500] + '...'

        from qtpy import QtWidgets
        from qtpy.QtCore import Qt

        qmb = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "Error", m)
        qmb.setDetailedText(detail)
        qmb.resize(400, qmb.size().height())
        qmb.setTextInteractionFlags(Qt.TextSelectableByMouse)
        qmb.exec_()

        if self.exit:
            sys.exit(1)

        # Just for cases where we are testing and patching sys.exit
        return True


class die_on_error(messagebox_on_error):

    def __init__(self, msg):
        super(die_on_error, self).__init__(msg, exit=True)
