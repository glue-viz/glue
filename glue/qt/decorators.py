from functools import wraps


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
