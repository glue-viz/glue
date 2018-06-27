from __future__ import absolute_import, division, print_function

import weakref
from itertools import chain
from weakref import WeakKeyDictionary
from contextlib import contextmanager

from .callback_container import CallbackContainer

__all__ = ['CallbackProperty', 'callback_property',
           'add_callback', 'remove_callback',
           'delay_callback', 'ignore_callback',
           'HasCallbackProperties', 'keep_in_sync']


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

    def _get_full_info(self, instance):

        # Some callback subclasses may contain additional info in addition
        # to the main value, and we need to use this full information when
        # comparing old and new 'values', so this method is used in that
        # case. The result should be a tuple where the first item is the
        # actual primary value of the property and the second item is any
        # additional data to use in the comparison.

        # Note that we need to make sure we convert any list here to a tuple
        # to make sure the value is immutable, otherwise comparisons of
        # old != new will not show any difference (since the list can still)
        # be modified in-place

        value = self.__get__(instance)
        if isinstance(value, list):
            value = tuple(value)
        return value, None

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
        if not self.enabled(instance):
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

    def enabled(self, instance):
        return not self._disabled.get(instance, False)

    def add_callback(self, instance, func, echo_old=False, priority=0):
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
        priority : int, optional
            This can optionally be used to force a certain order of execution of
            callbacks (larger values indicate a higher priority).
        """

        if echo_old:
            self._2arg_callbacks.setdefault(instance, CallbackContainer()).append(func, priority=priority)
        else:
            self._callbacks.setdefault(instance, CallbackContainer()).append(func, priority=priority)

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
            if func in cb[instance]:
                cb[instance].remove(func)
                return
        else:
            raise ValueError("Callback function not found: %s" % func)

    def clear_callbacks(self, instance):
        """
        Remove all callbacks on this property.
        """
        for cb in [self._callbacks, self._2arg_callbacks]:
            if instance in cb:
                cb[instance].clear()
        if instance in self._disabled:
            self._disabled.pop(instance)


class HasCallbackProperties(object):
    """
    A class that adds functionality to subclasses that use callback properties.
    """

    def __init__(self):
        from .list import ListCallbackProperty
        self._global_callbacks = CallbackContainer()
        self._ignored_properties = set()
        self._delayed_properties = {}
        self._delay_global_calls = {}
        self._callback_wrappers = {}
        for prop_name, prop in self.iter_callback_properties():
            if isinstance(prop, ListCallbackProperty):
                prop.add_callback(self, self._notify_global_lists)

    def _ignore_global_callbacks(self, properties):
        # This is to allow ignore_callbacks to work for global callbacks
        self._ignored_properties.update(properties)

    def _unignore_global_callbacks(self, properties):
        # Once this is called, we simply remove properties from _ignored_properties
        # and don't call the callbacks. This is used by ignore_callback
        self._ignored_properties -= set(properties)

    def _delay_global_callbacks(self, properties):
        # This is to allow delay_callback to still have an effect in delaying
        # global callbacks. We set _delayed_properties to a dictionary of the
        # values at the point at which the callbacks are delayed.
        self._delayed_properties.update(properties)

    def _process_delayed_global_callbacks(self, properties):
        # Once this is called, the global callbacks are called once each with
        # a dictionary of the current values of properties that have been
        # resumed.
        kwargs = {}
        for prop, new_value in properties.items():
            old_value = self._delayed_properties.pop(prop)
            if old_value != new_value:
                kwargs[prop] = new_value[0]
        self._notify_global(**kwargs)

    def _notify_global_lists(self, *args):
        from .list import ListCallbackProperty
        properties = {}
        for prop_name, prop in self.iter_callback_properties():
            if isinstance(prop, ListCallbackProperty):
                callback_list = getattr(self, prop_name)
                if callback_list is args[0]:
                    properties[prop_name] = callback_list
                    break
        self._notify_global(**properties)

    def _notify_global(self, **kwargs):
        for prop in set(self._delayed_properties) | set(self._ignored_properties):
            if prop in kwargs:
                kwargs.pop(prop)
        if len(kwargs) > 0:
            for callback in self._global_callbacks:
                callback(**kwargs)

    def __setattr__(self, attribute, value):
        super(HasCallbackProperties, self).__setattr__(attribute, value)
        if self.is_callback_property(attribute):
            self._notify_global(**{attribute: value})

    def add_callback(self, name, callback, echo_old=False, priority=0):
        """
        Add a callback that gets triggered when a callback property of the
        class changes.

        Parameters
        ----------
        name : str
            The instance to add the callback to.
        callback : func
            The callback function to add
        echo_old : bool, optional
            If `True`, the callback function will be invoked with both the old
            and new values of the property, as ``callback(old, new)``. If `False`
            (the default), will be invoked as ``callback(new)``
        priority : int, optional
            This can optionally be used to force a certain order of execution of
            callbacks (larger values indicate a higher priority).
        """
        if self.is_callback_property(name):
            prop = getattr(type(self), name)
            prop.add_callback(self, callback, echo_old=echo_old, priority=priority)
        else:
            raise TypeError("attribute '{0}' is not a callback property".format(name))

    def remove_callback(self, name, callback):
        """
        Remove a previously-added callback

        Parameters
        ----------
        name : str
            The instance to remove the callback from.
        func : func
            The callback function to remove
        """

        if self.is_callback_property(name):
            prop = getattr(type(self), name)
            try:
                prop.remove_callback(self, callback)
            except ValueError:  # pragma: nocover
                pass  # Be forgiving if callback was already removed before
        else:
            raise TypeError("attribute '{0}' is not a callback property".format(name))

    def add_global_callback(self, callback):
        """
        Add a global callback function, which is a callback that gets triggered
        when any callback properties on the class change.

        Parameters
        ----------
        callback : func
            The callback function to add
        """
        self._global_callbacks.append(callback)

    def remove_global_callback(self, callback):
        """
        Remove a global callback function.

        Parameters
        ----------
        callback : func
            The callback function to remove
        """
        self._global_callbacks.remove(callback)

    def is_callback_property(self, name):
        """
        Whether a property (identified by name) is a callback property.

        Parameters
        ----------
        name : str
            The name of the property to check
        """
        return isinstance(getattr(type(self), name, None), CallbackProperty)

    def iter_callback_properties(self):
        """
        Iterator to loop over all callback properties.
        """
        for name in dir(self):
            if self.is_callback_property(name):
                yield name, getattr(type(self), name)

    def clear_callbacks(self):
        """
        Remove all global and property-specific callbacks.
        """
        self._global_callbacks.clear()
        for name, prop in self.iter_callback_properties():
            prop.clear_callbacks(self)


def add_callback(instance, prop, callback, echo_old=False, priority=0):
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
    priority : int, optional
        This can optionally be used to force a certain order of execution of
        callbacks (larger values indicate a higher priority).

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
    p.add_callback(instance, callback, echo_old=echo_old, priority=priority)


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

        delay_props = {}

        for prop in self.props:

            p = getattr(type(self.instance), prop)
            if not isinstance(p, CallbackProperty):
                raise TypeError("%s is not a CallbackProperty" % prop)

            if (self.instance, prop) not in self.delay_count:
                self.delay_count[self.instance, prop] = 1
                self.old_values[self.instance, prop] = p._get_full_info(self.instance)
                delay_props[prop] = p._get_full_info(self.instance)
            else:
                self.delay_count[self.instance, prop] += 1

            p.disable(self.instance)

        if isinstance(self.instance, HasCallbackProperties):
            self.instance._delay_global_callbacks(delay_props)

    def __exit__(self, *args):

        resume_props = {}

        notifications = []

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
                new = p._get_full_info(self.instance)
                if old != new:
                    notifications.append((p, (self.instance, old[0], new[0])))
                resume_props[prop] = new

        if isinstance(self.instance, HasCallbackProperties):
            self.instance._process_delayed_global_callbacks(resume_props)

        for p, args in notifications:
            p.notify(*args)


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

    if isinstance(instance, HasCallbackProperties):
        instance._ignore_global_callbacks(props)

    yield

    for prop in props:
        p = getattr(type(instance), prop)
        assert isinstance(p, CallbackProperty)
        p.enable(instance)

    if isinstance(instance, HasCallbackProperties):
        instance._unignore_global_callbacks(props)


class keep_in_sync(object):

    def __init__(self, instance1, prop1, instance2, prop2):

        self.instance1 = weakref.ref(instance1, self.disable_syncing)
        self.prop1 = prop1

        self.instance2 = weakref.ref(instance2, self.disable_syncing)
        self.prop2 = prop2

        self._syncing = False

        self.enabled = False

        self.enable_syncing()

    def prop1_from_prop2(self, value):
        if not self._syncing:
            self._syncing = True
            setattr(self.instance1(), self.prop1, getattr(self.instance2(), self.prop2))
            self._syncing = False

    def prop2_from_prop1(self, value):
        if not self._syncing:
            self._syncing = True
            setattr(self.instance2(), self.prop2, getattr(self.instance1(), self.prop1))
            self._syncing = False

    def enable_syncing(self, *args):
        if self.enabled:
            return
        add_callback(self.instance1(), self.prop1, self.prop2_from_prop1)
        add_callback(self.instance2(), self.prop2, self.prop1_from_prop2)
        self.enabled = True

    def disable_syncing(self, *args):
        if not self.enabled:
            return
        if self.instance1() is not None:
            remove_callback(self.instance1(), self.prop1, self.prop2_from_prop1)
        if self.instance2() is not None:
            remove_callback(self.instance2(), self.prop2, self.prop1_from_prop2)
        self.enabled = False
