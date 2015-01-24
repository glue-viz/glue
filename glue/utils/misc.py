from __future__ import absolute_import, division, print_function

import string
from functools import partial

__all__ = ['DeferredMethod', 'nonpartial', 'lookup_class', 'as_variable_name',
           'as_list']


class DeferredMethod(object):
    """
    This class stubs out a method, and provides a
    callable interface that logs its calls. These
    can later be actually executed on the original (non-stubbed)
    method by calling executed_deferred_calls
    """

    def __init__(self, method):
        self.method = method
        self.calls = []  # avoid hashability issues with dict/set

    @property
    def original_method(self):
        return self.method

    def __call__(self, instance, *a, **k):
        if instance not in (c[0] for c in self.calls):
            self.calls.append((instance, a, k))

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self.__call__, instance)

    def execute_deferred_calls(self):
        for instance, args, kwargs in self.calls:
            self.method(instance, *args, **kwargs)


def nonpartial(func, *args, **kwargs):
    """Like functools.partial, this returns a function which,
    when called, calls func(*args, **kwargs). Unlike functools.partial,
    extra arguments passed to the returned function are *not* passed
    to the input function.

    This is used when connecting slots to QAction.triggered signals,
    which appear to have different signatures, which seem to add
    and extra argument in PyQt4 but not PySide
    """
    def result(*a, **k):
        return func(*args, **kwargs)

    return result


def lookup_class(ref):
    """ Look up an object via it's module string (e.g., 'glue.core.Data')

    :param ref: reference
    :type ref: str
    :rtype: object, or None if not found
    """
    mod = ref.split('.')[0]
    try:
        result = __import__(mod)
    except ImportError:
        return None
    try:
        for attr in ref.split('.')[1:]:
            result = getattr(result, attr)
        return result
    except AttributeError:
        return None


def as_variable_name(x):
    """
    Convert a string to a legal python variable name

    :param x: A string to (possibly) rename
    :returns: A legal python variable name
    """
    allowed = string.ascii_letters + string.digits + '_'
    result = [letter if letter in allowed else '_' for letter in x or 'x']
    if result[0] in string.digits:
        result.insert(0, '_')
    return ''.join(result)


def as_list(x):
    if isinstance(x, list):
        return x
    return [x]
