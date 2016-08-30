from __future__ import absolute_import, division, print_function

import traceback

__all__ = ['die_on_error']


def die_on_error(msg):
    """
    Non-GUI version of the decorator in glue.utils.qt.decorators.

    In this case we just let the Python exception terminate the execution.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print('=' * 72)
                print(msg + ' (traceback below)')
                print('-' * 72)
                traceback.print_exc()
                print('=' * 72)
        return wrapper
    return decorator
