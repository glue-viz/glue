from collections import defaultdict
from contextlib import contextmanager


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
        self._disabled = defaultdict(bool)
        self._values = dict()

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        return self._values.get(instance, self._default)

    def __set__(self, instance, value):
        changed = self.__get__(instance) != value
        self._values[instance] = value
        if changed:
            self.notify(instance)

    def notify(self, instance):
        """Call all callback functions with the current value

        :param instance: The instance to consider
        """
        if self._disabled[instance]:
            return
        value = self.__get__(instance)
        for cback in self._callbacks[instance]:
            cback(value)

    def disable(self, instance):
        """Disable callbacks for a specific instance"""
        self._disabled[instance] = True

    def enable(self, instance):
        """Enable previously-disabled callbacks for a specific instance"""
        self._disabled[instance] = False

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


@contextmanager
def delay_callback(instance, *props):
    """Delay any callbacks from a callback property

    This is a context manager. Within the context block, no callbacks
    will be issued. Each callback will be called once on exit

    :param instance: An instance object with CallbackProperties

    :param props: One or more properties within instance to delay
    :type prop: str
    """
    vals = []
    for prop in props:
        p = getattr(type(instance), prop)
        if not isinstance(p, CallbackProperty):
            raise TypeError("%s is not a CallbackProperty" % prop)
        vals.append(p.__get__(instance))
        p.disable(instance)

    yield

    for v, prop in zip(vals, props):
        p = getattr(type(instance), prop)
        assert isinstance(p, CallbackProperty)
        p.enable(instance)
        if p.__get__(instance) != v:
            p.notify(instance)


@contextmanager
def ignore_callback(instance, *props):
    """Temporarily ignore any callbacks from a callback property

    This is a context manager. Within the context block, no callbacks
    will be issued. In contrast with delay_callback, no callbakcs
    will be called on exiting the context manager

    :param instance: An instance object with CallbackProperties

    :param props: One or more properties within instance to delay
    :type prop: str
    """
    for prop in props:
        p = getattr(type(instance), prop)
        if not isinstance(p, CallbackProperty):
            raise TypeError("%s is not a CallbackProperty" % prop)
        p.disable(instance)

    yield

    for prop in props:
        p = getattr(type(instance), prop)
        assert isinstance(p, CallbackProperty)
        p.enable(instance)
