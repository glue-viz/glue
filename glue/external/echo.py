from contextlib import contextmanager
from weakref import WeakKeyDictionary


__all__ = ['CallbackProperty', 'callback_property',
           'add_callback', 'remove_callback',
           'delay_callback', 'ignore_callback']


class CallbackProperty(object):

    """A property that callback functions can be added to.

    When a callback property changes value, each callback function
    is called with information about the state change. Otherwise,
    callback properties behave just like normal instance variables.

    CallbackProperties must be defined at the class level. Use
    the helper function :func:`add_callback` to attach a callback to
    a specific instance of a class with CallbackProperties
    """

    def __init__(self, default=None, getter=None, setter=None):
        """
        :param default: The initial value for the property
        """
        self._default = default
        self._callbacks = WeakKeyDictionary()
        self._2arg_callbacks = WeakKeyDictionary()
        self._disabled = WeakKeyDictionary()
        self._values = WeakKeyDictionary()

        if getter is None:
            getter = self._default_getter

        if setter is None:
            setter = self._default_setter

        self._getter = getter
        self._setter = setter

    def _default_getter(self, instance, owner=None):
        return self._values.get(instance, self._default)

    def _default_setter(self, instance, value):
        self._values.__setitem__(instance, value)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self._getter(instance)

    def __set__(self, instance, value):
        old = self.__get__(instance)
        self._setter(instance, value)
        new = self.__get__(instance)
        if old != new:
            self.notify(instance, old, new)

    def setter(self, func):
        """
        Method to use as a decorator, to mimic @property.setter
        """
        self._setter = func
        return self

    def notify(self, instance, old, new):
        """Call all callback functions with the current value

        :param instance: The instance to consider
        :param old: The old value of the property
        :param new: The new value of the property

        Each callback will either be called using
        callback(new) or callback(old, new) depending
        on whether echo_old was True during add_callback
        """
        if self._disabled.get(instance, False):
            return
        for cback in self._callbacks.get(instance, []):
            cback(new)
        for cback in self._2arg_callbacks.get(instance, []):
            cback(old, new)

    def disable(self, instance):
        """Disable callbacks for a specific instance"""
        self._disabled[instance] = True

    def enable(self, instance):
        """Enable previously-disabled callbacks for a specific instance"""
        self._disabled[instance] = False

    def add_callback(self, instance, func, echo_old=False):
        """Add a callback to a specific instance that manages this property

        :param instance: Instance to bind the callback to
        :param func: Callback function
        :param echo_old: If true, the callback function will be invoked
        with both the old and new values of the property, as func(old, new)
        If False (the default), will be invoked as func(new)
        """
        if echo_old:
            self._2arg_callbacks.setdefault(instance, []).append(func)
        else:
            self._callbacks.setdefault(instance, []).append(func)

    def remove_callback(self, instance, func):
        """Remove a previously-added callback

        :param instance: The instance to detach the callback from
        :param func: The callback function to remove
        """
        for cb in [self._callbacks, self._2arg_callbacks]:
            if instance not in cb:
                continue
            try:
                cb[instance].remove(func)
                break
            except ValueError:
                pass
        else:
            raise ValueError("Callback function not found: %s" % func)


def add_callback(instance, prop, callback, echo_old=False):
    """Attach a callback function to a property in an instance

    :param instance: Instance of a class with callback properties

    :param prop: Name of callback property in `instance`
    :type prop: str

    :param callback: Callback function
    :type callback: Callable

    Example::

        class Foo:
            bar = CallbackProperty(0)

        def callback(value):
            pass

        f = Foo()
        add_callback(f, 'bar', callback)

    """
    p = getattr(type(instance), prop)
    if not isinstance(p, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop)
    p.add_callback(instance, callback, echo_old=echo_old)


def remove_callback(instance, prop, callback):
    """Remove a callback function from a property in an instance

    :param instance: Instance of a class with callback properties

    :param prop: Name of callback property in `instance`
    :type prop: str

    :param callback: Callback function
    :type callback: Callable
    """
    p = getattr(type(instance), prop)
    if not isinstance(p, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop)
    p.remove_callback(instance, callback)


def callback_property(getter):
    """
    A decorator to build a CallbackProperty,
    by wrapping a getter method, similar to the use
    of @property::

        class Foo(object):
            @callback_property
            def x(self):
                 return self._x

            @x.setter
            def x(self, value):
                self._x = value

    In simple cases with no getter or setter logic, it's
    easier to create a CallbackProperty directly::

        class Foo(object);
            x = CallbackProperty(initial_value)
    """
    return CallbackProperty(getter=getter)


@contextmanager
def delay_callback(instance, *props):
    """Delay any callback functions from one or more callback properties

    This is a context manager. Within the context block, no callbacks
    will be issued. Each callback will be called once on exit

    :param instance: An instance object with CallbackProperties

    :param props: One or more properties within instance to delay
    :type prop: str

    Example::

        with delay_callback(foo, 'bar', 'baz'):
            f.bar = 20
            f.baz = 30
            f.bar = 10
        print 'done'  # callbacks triggered at this point, if needed
    """
    vals = []
    for prop in props:
        p = getattr(type(instance), prop)
        if not isinstance(p, CallbackProperty):
            raise TypeError("%s is not a CallbackProperty" % prop)
        vals.append(p.__get__(instance))
        p.disable(instance)

    yield

    for old, prop in zip(vals, props):
        p = getattr(type(instance), prop)
        assert isinstance(p, CallbackProperty)
        p.enable(instance)
        new = p.__get__(instance)
        if old != new:
            p.notify(instance, old, new)


@contextmanager
def ignore_callback(instance, *props):
    """Temporarily ignore any callbacks from one or more callback properties

    This is a context manager. Within the context block, no callbacks
    will be issued. In contrast with delay_callback, no callbakcs
    will be called on exiting the context manager

    :param instance: An instance object with CallbackProperties

    :param props: One or more properties within instance to delay
    :type prop: str

    Example::

        with ignore_callback(foo, 'bar', 'baz'):
                f.bar = 20
                f.baz = 30
                f.bar = 10
        print 'done'  # no callbacks called

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
