import queue
import string
from functools import partial, reduce

from echo.callback_container import CallbackContainer


__all__ = ['DeferredMethod', 'nonpartial', 'lookup_class', 'as_variable_name',
           'as_list', 'file_format', 'CallbackMixin', 'PropertySetMixin',
           'Pointer', 'common_prefix', 'queue_to_list', 'format_choices']


class DeferredMethod(object):
    """
    This class stubs out a method, and provides a callable interface that logs
    its calls. These can later be actually executed on the original
    (non-stubbed) method by calling executed_deferred_calls
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
    """
    Like functools.partial, this returns a function which, when called, calls
    ``func(*args, **kwargs)``.

    Unlike functools.partial, extra arguments passed to the returned function
    are *not* passed to the input function.

    This is used when connecting slots to ``QAction.triggered`` signals, which
    appear to have different signatures, which seem to add and extra argument
    in PyQt but not PySide
    """
    def result(*a, **k):
        return func(*args, **kwargs)

    return result


def lookup_class(ref):
    """
    Look up an object via its module string (e.g., 'glue.core.data.Data')

    Parameters
    ----------
    ref : str
        The module string
    """

    if ref.startswith('__builtin__'):
        ref = '.'.join(['builtins'] + ref.split('.')[1:])

    mod = ref.rsplit('.', 1)[0]

    try:
        result = __import__(mod)
    except ImportError:
        raise ValueError("Module '{0}' not found".format(mod))

    try:
        for attr in ref.split('.')[1:]:
            result = getattr(result, attr)
        return result
    except AttributeError:
        raise ValueError("Object '{0}' not found".format(ref))


def as_variable_name(x):
    """
    Convert a string to a legal python variable name

    Parameters
    ----------
    x : str
        A string to (possibly) rename

    Returns
    -------
    variable_name : str
        A legal Python variable name
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


def file_format(filename):
    if filename.find('.') == -1:
        return ''
    if filename.lower().endswith('.gz'):
        result = filename.lower().rsplit('.', 2)[1]
    else:
        result = filename.lower().rsplit('.', 1)[1]
    return result


class CallbackMixin(object):

    """
    A mixin that provides a utility for attaching callback
    functions to methods
    """

    def __init__(self):
        self._callbacks = CallbackContainer()

    def add_callback(self, function):
        self._callbacks.append(function)

    def remove_callback(self, function):
        self._callbacks.remove(function)

    def notify(self, *args, **kwargs):
        for func in self._callbacks:
            func(*args, **kwargs)


class PropertySetMixin(object):

    """An object that provides a set of properties that
    are meant to encapsulate state information

    This class exposes a properties attribute, which is a dict
    of all properties. Similarly, assigning to the properties dict
    will update the individual properties
    """
    _property_set = []

    @property
    def properties(self):
        """ A dict mapping property names to values """
        return dict((p, getattr(self, p)) for p in self._property_set)

    @properties.setter
    def properties(self, value):
        """ Update the properties with a new dict.

        Keys in the new dict must be valid property names defined in
        the _property_set class level attribute"""
        invalid = set(value.keys()) - set(self._property_set)
        if invalid:
            raise ValueError("Invalid property values: %s" % invalid)

        for k in self._property_set:
            if k not in value:
                continue
            setattr(self, k, value[k])


def common_prefix(strings, exclude_punctuation=True):
    """
    Given a list of strings, find the longest prefix common to all of them
    """
    if len(strings) > 0:
        for i in range(len(strings[0]), 0, -1):
            if exclude_punctuation and strings[0][i - 1] in string.punctuation:
                continue
            for st in strings[1:]:
                if st[:i] != strings[0][:i]:
                    break
            else:
                return strings[0][:i]
    return ''


class Pointer(object):

    def __init__(self, key):
        self.key = key

    def __get__(self, instance, type=None):
        val = instance
        for k in self.key.split('.'):
            val = getattr(val, k, None)
        return val

    def __set__(self, instance, value):
        v = self.key.split('.')
        attr = reduce(getattr, [instance] + v[:-1])
        setattr(attr, v[-1], value)


def queue_to_list(q):
    """
    Get all the values in a :class:`queue.Queue` object and return a list.
    """
    l = []
    while True:
        try:
            l.append(q.get_nowait())
        except queue.Empty:
            return l


def format_choices(options, index=False):
    """
    Return a string with an error message formatted as:

    * option 1
    * option 2

    This can be preprended to existing error messages.
    """
    updated_options = []
    for option in options:
        if isinstance(option, str):
            updated_options.append("'{0}'".format(option))
        elif isinstance(option, type):
            updated_options.append(str(option.__module__) + '.' + option.__name__)
        else:
            updated_options.append(option)
    if index:
        return "\n\n" + '\n'.join(['* {0} or {1}'.format(idx, option) for idx, option in enumerate(updated_options)])
    else:
        return "\n\n" + '\n'.join(['* {0}'.format(option) for option in updated_options])
