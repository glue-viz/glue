import inspect
import traceback

__all__ = ['die_on_error', 'avoid_circular', 'decorate_all_methods']


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


def avoid_circular(meth):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_in_avoid_circular') or not self._in_avoid_circular:
            self._in_avoid_circular = True
            try:
                return meth(self, *args, **kwargs)
            finally:
                self._in_avoid_circular = False
    return wrapper


def decorate_all_methods(decorator):

    def decorate(cls):
        for name, value in inspect.getmembers(cls, inspect.isfunction):
            setattr(cls, name, decorator(value))
        return cls

    return decorate
