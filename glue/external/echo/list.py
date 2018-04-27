import sys

from . import CallbackProperty, HasCallbackProperties


class CallbackList(list):
    """
    A list that calls a callback function when it is modified.

    The first argument should be the callback function (which takes no
    arguments), and subsequent arguments are as for `list`.
    """

    def __init__(self, callback, *args, **kwargs):
        super(CallbackList, self).__init__(*args, **kwargs)
        self.callback = callback

    def append(self, value):
        super(CallbackList, self).append(value)
        if isinstance(value, HasCallbackProperties):
            value.add_global_callback(self.callback)
        self.callback()

    def extend(self, iterable):
        super(CallbackList, self).extend(iterable)
        for item in iterable:
            if isinstance(item, HasCallbackProperties):
                item.add_global_callback(self.callback)
        self.callback()

    def insert(self, index, value):
        super(CallbackList, self).insert(index, value)
        if isinstance(value, HasCallbackProperties):
            value.add_global_callback(self.callback)
        self.callback()

    def pop(self, index=-1):
        result = super(CallbackList, self).pop(index)
        if isinstance(result, HasCallbackProperties):
            result.remove_global_callback(self.callback)
        self.callback()
        return result

    def remove(self, value):
        if isinstance(value, HasCallbackProperties):
            value.remove_global_callback(self.callback)
        super(CallbackList, self).remove(value)
        self.callback()

    def reverse(self):
        super(CallbackList, self).reverse()
        self.callback()

    def sort(self, key=None, reverse=False):
        super(CallbackList, self).sort(key=key, reverse=reverse)
        self.callback()

    def __setitem__(self, slc, new_value):

        old_values = self[slc]
        if not isinstance(slc, slice):
            old_values = [old_values]

        for old_value in old_values:
            if isinstance(old_value, HasCallbackProperties):
                old_value.remove_global_callback(self.callback)

        if isinstance(slc, slice):
            new_values = new_value
        else:
            new_values = [new_value]

        for value in new_values:
            if isinstance(value, HasCallbackProperties):
                value.add_global_callback(self.callback)

        super(CallbackList, self).__setitem__(slc, new_value)
        self.callback()

    if sys.version_info[0] >= 3:

        def clear(self):
            for item in self:
                if isinstance(item, HasCallbackProperties):
                    item.remove_global_callback(self.callback)
            super(CallbackList, self).clear()
            self.callback()

    else:

        def __setslice__(self, start, end, new_values):

            slc = slice(start, end)

            old_values = self[slc]

            for old_value in old_values:
                if isinstance(old_value, HasCallbackProperties):
                    old_value.remove_global_callback(self.callback)

            for value in new_values:
                if isinstance(value, HasCallbackProperties):
                    value.add_global_callback(self.callback)

            super(CallbackList, self).__setslice__(start, end, new_values)
            self.callback()


class ListCallbackProperty(CallbackProperty):
    """
    A list property that calls callbacks when its contents are modified
    """

    def _default_getter(self, instance, owner=None):
        if instance not in self._values:
            self._default_setter(instance, [])
        return super(ListCallbackProperty, self)._default_getter(instance, owner)

    def _default_setter(self, instance, value):

        if not isinstance(value, list):
            raise TypeError('callback property should be a list')

        def callback(*args, **kwargs):
            self.notify(instance, wrapped_list, wrapped_list)

        wrapped_list = CallbackList(callback, value)
        super(ListCallbackProperty, self)._default_setter(instance, wrapped_list)
