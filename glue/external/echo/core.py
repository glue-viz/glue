from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
from weakref import WeakKeyDictionary
from functools import partial

__all__ = ['CallbackProperty', 'callback_property',
           'add_callback', 'remove_callback',
           'delay_callback', 'ignore_callback',
           'HasCallbackProperties']


class CallbackProperty(object):
    """
    A property that callback functions can be added to.

    When a callback property changes value, each callback function
    is called with information about the state change. Otherwise,
    callback properties behave just like normal instance variables.

    CallbackProperties must be defined at the class level. Use
    the helper function :func:`~echo.add_callback` to attach a callback to
    a specific instance of a class with CallbackProperties

    Parameters
    ----------
    default
        The initial value for the property
    docstring : str
        The docstring for the property
    getter, setter : func
        Custom getter and setter functions (advanced)
    """

    def __init__(self, default=None, docstring=None, getter=None, setter=None):
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

        if docstring is not None:
            self.__doc__ = docstring

    def _default_getter(self, instance, owner=None):
        return self._values.get(instance, self._default)

    def _default_setter(self, instance, value):
        self._values.__setitem__(instance, value)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self._getter(instance)

    def __set__(self, instance, value):
        try:
            old = self.__get__(instance)
        except AttributeError:  # pragma: no cover
            old = None
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
        """
        Call all callback functions with the current value

        Each callback will either be called using
        callback(new) or callback(old, new) depending
        on whether ``echo_old`` was set to `True` when calling
        :func:`~echo.add_callback`

        Parameters
        ----------
        instance
            The instance to consider
        old
            The old value of the property
        new
            The new value of the property
        """
        if self._disabled.get(instance, False):
            return
        for cback in self._callbacks.get(instance, []):
            cback(new)
        for cback in self._2arg_callbacks.get(instance, []):
            cback(old, new)

    def disable(self, instance):
        """
        Disable callbacks for a specific instance
        """
        self._disabled[instance] = True

    def enable(self, instance):
        """
        Enable previously-disabled callbacks for a specific instance
        """
        self._disabled[instance] = False

    def add_callback(self, instance, func, echo_old=False):
        """
        Add a callback to a specific instance that manages this property

        Parameters
        ----------
        instance
            The instance to add the callback to
        func : func
            The callback function to add
        echo_old : bool, optional
            If `True`, the callback function will be invoked with both the old
            and new values of the property, as ``func(old, new)``. If `False`
            (the default), will be invoked as ``func(new)``
        """
        if echo_old:
            self._2arg_callbacks.setdefault(instance, []).append(func)
        else:
            self._callbacks.setdefault(instance, []).append(func)

    def remove_callback(self, instance, func):
        """
        Remove a previously-added callback

        Parameters
        ----------
        instance
            The instance to detach the callback from
        func : func
            The callback function to remove
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


class HasCallbackProperties(object):
    """
    A class that adds functionality to subclasses that use callback properties.
    """

    def __init__(self):
        self._global_callbacks = []
        self._callback_wrappers = {}

    def add_callback(self, name, callback, echo_old=False, as_kwargs=False):
        """
        Add a callback that gets triggered when a callback property of the
        class changes.

        Parameters
        ----------
        name : str
            The instance to add the callback to. This can be ``'*'`` to
            indicate that the callback should be added to all callback
            properties.
        func : func
            The callback function to add
        echo_old : bool, optional
            If `True`, the callback function will be invoked with both the old
            and new values of the property, as ``func(old, new)``. If `False`
            (the default), will be invoked as ``func(new)``
        as_kwargs : bool, optional
            If `True`, the callback function will be invoked using keyword arguments
            where the keyword is the name of the attribute, and the value is either
            the new value or a tuple of (old, new) if echo_old is `True`.
        """

        if name == '*':
            for prop_name, prop in self.iter_callback_properties():
                self.add_callback(prop_name, callback, echo_old=echo_old, as_kwargs=as_kwargs)
        else:
            if self.is_callback_property(name):
                if as_kwargs:
                    def wrap_callback(function, name):
                        def callback_wrapper(value):
                            return function(**{name: value})
                        return callback_wrapper
                    self._callback_wrappers[(name, callback)] = wrap_callback(callback, name)
                    callback = self._callback_wrappers[(name, callback)]
                prop = getattr(type(self), name)
                prop.add_callback(self, callback, echo_old=echo_old)
            else:
                raise TypeError("attribute '{0}' is not a callback property".format(name))

    def remove_callback(self, name, callback):
        """
        Remove a previously-added callback

        Parameters
        ----------
        name : str
            The instance to remove the callback from. This can be ``'*'`` to
            indicate that the callback should be removed from all callback
            properties.
        func : func
            The callback function to remove
        """

        if name == '*':
            for prop_name, prop in self.iter_callback_properties():
                self.remove_callback(prop_name, callback)
        else:
            if self.is_callback_property(name):
                if (name, callback) in self._callback_wrappers:
                    callback = self._callback_wrappers.pop((name, callback))
                prop = getattr(type(self), name)
                try:
                    prop.remove_callback(self, callback)
                except ValueError:
                    pass  # Be forgiving if callback was already removed before
            else:
                raise TypeError("attribute '{0}' is not a callback property".format(name))

    def is_callback_property(self, name):
        return isinstance(getattr(type(self), name, None), CallbackProperty)

    def iter_callback_properties(self):
        for name in dir(self):
            if self.is_callback_property(name):
                yield name, getattr(type(self), name)


def add_callback(instance, prop, callback, echo_old=False):
    """
    Attach a callback function to a property in an instance

    Parameters
    ----------
    instance
        The instance to add the callback to
    prop : str
        Name of callback property in `instance`
    callback : func
        The callback function to add
    echo_old : bool, optional
        If `True`, the callback function will be invoked with both the old
        and new values of the property, as ``func(old, new)``. If `False`
        (the default), will be invoked as ``func(new)``

    Examples
    --------

    ::

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
    """
    Remove a callback function from a property in an instance

    Parameters
    ----------
    instance
        The instance to detach the callback from
    prop : str
        Name of callback property in `instance`
    callback : func
        The callback function to remove
    """
    p = getattr(type(instance), prop)
    if not isinstance(p, CallbackProperty):
        raise TypeError("%s is not a CallbackProperty" % prop)
    p.remove_callback(instance, callback)


def callback_property(getter):
    """
    A decorator to build a CallbackProperty.

    This is used by wrapping a getter method, similar to the use of @property::

        class Foo(object):
            @callback_property
            def x(self):
                 return self._x

            @x.setter
            def x(self, value):
                self._x = value

    In simple cases with no getter or setter logic, it's easier to create a
    :class:`~echo.CallbackProperty` directly::

        class Foo(object);
            x = CallbackProperty(initial_value)
    """

    cb = CallbackProperty(getter=getter)
    cb.__doc__ = getter.__doc__
    return cb


class delay_callback(object):
    """
    Delay any callback functions from one or more callback properties

    This is a context manager. Within the context block, no callbacks
    will be issued. Each callback will be called once on exit

    Parameters
    ----------
    instance
        An instance object with callback properties
    *props : str
        One or more properties within instance to delay

    Examples
    --------

    ::

        with delay_callback(foo, 'bar', 'baz'):
            f.bar = 20
            f.baz = 30
            f.bar = 10
        print('done')  # callbacks triggered at this point, if needed
    """

    # Class-level registry of properties and how many times the callbacks have
    # been delayed. The idea is that when nesting calls to delay_callback, the
    # delay count is increased, and every time __exit__ is called, the count is
    # decreased, and once the count reaches zero, the callback is triggered.
    delay_count = {}
    old_values = {}

    def __init__(self, instance, *props):
        self.instance = instance
        self.props = props

    def __enter__(self):

        for prop in self.props:

            p = getattr(type(self.instance), prop)
            if not isinstance(p, CallbackProperty):
                raise TypeError("%s is not a CallbackProperty" % prop)

            if (self.instance, prop) not in self.delay_count:
                self.delay_count[self.instance, prop] = 1
                self.old_values[self.instance, prop] = p.__get__(self.instance)
            else:
                self.delay_count[self.instance, prop] += 1

            p.disable(self.instance)

    def __exit__(self, *args):

        for prop in self.props:

            p = getattr(type(self.instance), prop)
            if not isinstance(p, CallbackProperty):  # pragma: no cover
                raise TypeError("%s is not a CallbackProperty" % prop)

            if self.delay_count[self.instance, prop] > 1:
                self.delay_count[self.instance, prop] -= 1
            else:
                self.delay_count.pop((self.instance, prop))
                old = self.old_values.pop((self.instance, prop))
                p.enable(self.instance)
                new = p.__get__(self.instance)
                if old != new:
                    p.notify(self.instance, old, new)


@contextmanager
def ignore_callback(instance, *props):
    """
    Temporarily ignore any callbacks from one or more callback properties

    This is a context manager. Within the context block, no callbacks will be
    issued. In contrast with :func:`~echo.delay_callback`, no callbakcs will be
    called on exiting the context manager

    Parameters
    ----------
    instance
        An instance object with callback properties
    *props : str
        One or more properties within instance to ignore

    Examples
    --------

    ::

        with ignore_callback(foo, 'bar', 'baz'):
                f.bar = 20
                f.baz = 30
                f.bar = 10
        print('done')  # no callbacks called

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
