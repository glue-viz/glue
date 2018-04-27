from __future__ import absolute_import, division, print_function

import weakref
from functools import partial

__all__ = ['HubCallbackContainer']


class HubCallbackContainer(object):
    """
    A list-like container for callback functions. We need to be careful with
    storing references to methods, because if a callback method is on a class
    which contains both the callback and a callback property, a circular
    reference is created which results in a memory leak. Instead, we need to use
    a weak reference which results in the callback being removed if the instance
    is destroyed. This container class takes care of this automatically.

    Adapted from echo.CallbackContainer.
    """

    def __init__(self):
        self.callbacks = {}

    def _wrap(self, handler, filter):
        """
        Given a function/method, this will automatically wrap a method using
        weakref to avoid circular references.
        """

        if not callable(handler):
            raise TypeError("Only callable handlers can be stored in CallbackContainer")

        if filter is not None and not callable(filter):
            raise TypeError("Only callable filters can be stored in CallbackContainer")

        if self.is_bound_method(handler):

            # We are dealing with a bound method. Method references aren't
            # persistent, so instead we store a reference to the function
            # and instance.

            value = (weakref.ref(handler.__func__),
                     weakref.ref(handler.__self__, self._auto_remove))

        else:

            value = (handler, None)

        if self.is_bound_method(filter):

            # We are dealing with a bound method. Method references aren't
            # persistent, so instead we store a reference to the function
            # and instance.

            value += (weakref.ref(filter.__func__),
                      weakref.ref(filter.__self__, self._auto_remove))

        else:

            value += (filter, None)

        return value

    def _auto_remove(self, method_instance):

        # Called when weakref detects that the instance on which a method was
        # defined has been garbage collected.
        remove = []
        for key, value in self.callbacks.items():
            if value[1] is method_instance or value[3] is method_instance:
                remove.append(key)
        for key in remove:
            self.callbacks.pop(key)

    def __contains__(self, message_class):
        return message_class in self.callbacks

    def __getitem__(self, message_class):

        callback = self.callbacks[message_class]

        if callback[1] is None:
            result = (callback[0],)
        else:
            func = callback[0]()
            inst = callback[1]()
            result = (partial(func, inst),)

        if callback[3] is None:
            result += (callback[2],)
        else:
            func = callback[2]()
            inst = callback[3]()
            result += (partial(func, inst),)

        return result

    def __iter__(self):
        for message_class in self.callbacks:
            yield self[message_class]

    def __len__(self):
        return len(self.callbacks)

    def keys(self):
        return self.callbacks.keys()

    @staticmethod
    def is_bound_method(func):
        return hasattr(func, '__func__') and getattr(func, '__self__', None) is not None

    def __setitem__(self, message_class, value):
        handler, filter = value
        self.callbacks[message_class] = self._wrap(handler, filter)

    def pop(self, message_class):
        return self.callbacks.pop(message_class)

    def remove_handler(self, handler):
        if self.is_bound_method(handler):
            for message_class in sorted(self.callbacks):
                callback = self.callbacks[message_class]
                if callback[1] is not None and handler.__func__ is callback[0]() and handler.__self__ is callback[1]():
                    self.callbacks.pop(callback)
        else:
            for message_class in sorted(self.callbacks):
                callback = self.callbacks[message_class]
                if callback[1] is None and handler is callback[0]:
                    self.callbacks.pop(callback)
