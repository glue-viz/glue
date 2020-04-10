from . import CallbackProperty, HasCallbackProperties


class CallbackDict(dict):
    """
    A dictionary that calls a callback function when it is modified.

    The first argument should be the callback function (which takes no
    arguments), and subsequent arguments are passed to `dict`.
    """

    def __init__(self, callback, *args, **kwargs):
        super(CallbackDict, self).__init__(*args, **kwargs)

        self.callback = callback

    def clear(self):
        super().clear()

        self.callback()

    def popitem(self):
        result = super().popitem()

        if isinstance(result, HasCallbackProperties):
            result.remove_global_callback(self.callback)

        self.callback()

        return result

    def setdefault(self, *args, **kwargs):
        key = super().setdefault(*args, **kwargs)

        if isinstance(key, HasCallbackProperties):
            key.remove_global_callback(self.callback)

        self.callback()

        return key

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)

        self.callback()

    def pop(self, *args, **kwargs):
        result = super().pop(*args, **kwargs)

        if isinstance(result, HasCallbackProperties):
            result.remove_global_callback(self.callback)

        self.callback()

        return result

    def __reversed__(self):
        result = super().__reversed__()

        self.callback()

        return result

    def __setitem__(self, k, v):
        super().__setitem__(k, v)

        self.callback()

    def __delattr__(self, *args):
        super().__delattr__(*args)

        self.callback()

    def __repr__(self):
        return f"<CallbackDict with {len(self)} elements>"


class DictCallbackProperty(CallbackProperty):
    """
    A dictionary property that calls callbacks when its contents are modified
    """
    def _default_getter(self, instance, owner=None):
        if instance not in self._values:
            self._default_setter(instance, {})
        return super()._default_getter(instance, owner)

    def _default_setter(self, instance, value):

        if not isinstance(value, dict):
            raise TypeError("Callback property should be a dictionary.")

        def callback(*args, **kwargs):
            self.notify(instance, wrapped_dict, wrapped_dict)

        wrapped_dict = CallbackDict(callback, value)

        super()._default_setter(instance, wrapped_dict)
