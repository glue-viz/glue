from collections import defaultdict


class CallbackProperty(object):
    """A property that callback functions can be added to.

    When a callback property changes value, each callback function
    is called with the changed value as the input argument. Otherwise,
    callback properties behave just like normal instance variables.

    CallbackProperties must be defined at the class-level. Use
    the helper function `add_callback` to attach a callback to
    a specific instance of a class with CallbackProperties


    Example:

        class Foo(object):
            x = CallbackProperty(5)

        def callback(value):
            print 'value changed to %s" % value

        f = Foo()
        add_callback(f, 'x', callback)
        f.x = 12  # 'value changed to 12'
    """
    def __init__(self, default=None):
        self._default = default
        self._callbacks = defaultdict(list)
        self._values = dict()

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        return self._values.get(instance, self._default)

    def __set__(self, instance, value):
        changed = self._values.get(instance, self._default) != value
        self._values[instance] = value
        if changed:
            for cback in self._callbacks[instance]:
                cback(value)

    def add_callback(self, instance, func):
        """Add a callback to a specific instance that manages this property

        It is more convenient to use the add_callback function than
        this method

        :param instance: Instance to bind the callback to
        :param func: Callback function
        """
        self._callbacks[instance].append(func)

    def remove_callback(self, instance, func):
        """Remove a previously-added callback

        :param instance: The instance to detach the callback from
        :param func: The callback function to remove
        """
        try:
            self._callbacks[instance].remove(func)
        except ValueError:
            raise ValueError("Callback function not found: %s" % func)


def add_callback(instance, prop, callback):
    """Attach a callback function to a property in an instance

    :param instance: Instance object

    :param prop: Name of property
    :type prop: str

    :param callback: Callback function
    :type callback: Callable
    """
    p = getattr(type(instance), prop)
    if not isinstance(p, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop)
    p.add_callback(instance, callback)


def remove_callback(instance, prop, callback):
    """Remove a callback function from a property in an instance

    :param instance: Instance object

    :param prop: Name of property
    :type prop: str

    :param callback: Callback function
    :type callback: Callable
    """
    p = getattr(type(instance), prop)
    if not isinstance(p, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop)
    p.remove_callback(instance, callback)
