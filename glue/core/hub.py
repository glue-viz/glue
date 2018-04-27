from __future__ import absolute_import, division, print_function

import logging
from contextlib import contextmanager
from weakref import WeakKeyDictionary
from inspect import getmro
from collections import Counter

from glue.core.exceptions import InvalidSubscriber, InvalidMessage
from glue.core.message import Message
from glue.core.hub_callback_container import HubCallbackContainer

__all__ = ['Hub', 'HubListener']


class Hub(object):

    """The hub manages communication between subscribers.

    Objects :func:`subscribe` to receive specific message types. When
    a message is passed to :func:`broadcast`, the hub observes the
    following protocol:

        * For each subscriber, it looks for a message class
          subscription that is a superclass of the input message type
          (if several are found, the most-subclassed one is chosen)

        * If one is found, it calls the subscriptions filter(message)
          class (if provided)

        * If filter(message) == True, it calls handler(message)
          (or notify(message) if handler wasn't provided).

    """

    def __init__(self, *args):
        """
        Any arguments that are passed to Hub will be registered
        to the new hub object.
        """
        # Dictionary of subscriptions
        self._subscriptions = WeakKeyDictionary()

        self._paused = False
        self._queue = []

        self._ignore = Counter()

        from glue.core.data import Data
        from glue.core.subset import Subset
        from glue.core.data_collection import DataCollection

        listeners = set(filter(lambda x: isinstance(x, HubListener), args))
        data = set(filter(lambda x: isinstance(x, Data), args))
        subsets = set(filter(lambda x: isinstance(x, Subset), args))
        dcs = set(filter(lambda x: isinstance(x, DataCollection), args))
        listeners -= (data | subsets | dcs)
        if set(listeners | data | subsets | dcs) != set(args):
            raise TypeError("Inputs must be HubListener, data, subset, or "
                            "data collection objects")

        for l in listeners:
            l.register_to_hub(self)
        for d in data:
            d.register_to_hub(self)
        for dc in dcs:
            dc.register_to_hub(self)
        for s in subsets:
            s.register()

    def subscribe(self, subscriber, message_class,
                  handler=None,
                  filter=lambda x: True):
        """Subscribe an object to a type of message class.

        :param subscriber: The subscribing object
        :type subscriber: :class:`~glue.core.hub.HubListener`

        :param message_class: A :class:`~glue.core.message.Message` class
                              to subscribe to

        :param handler:
           An optional function of the form handler(message) that will
           receive the message on behalf of the subscriber. If not provided,
           this defaults to the HubListener's notify method


        :param filter:
           An optional function of the form filter(message). Messages
           are only passed to the subscriber if filter(message) == True.
           The default is to always pass messages.


        Raises:
            InvalidMessage: If the input class isn't a
            :class:`~glue.core.message.Message` class

            InvalidSubscriber: If the input subscriber isn't a
            HubListener object.

        """
        if not isinstance(subscriber, HubListener):
            raise InvalidSubscriber("Subscriber must be a HubListener: %s" %
                                    type(subscriber))
        if not isinstance(message_class, type) or \
                not issubclass(message_class, Message):
            raise InvalidMessage("message class must be a subclass of "
                                 "glue.Message: %s" % type(message_class))
        logging.getLogger(__name__).info("Subscribing %s to %s",
                                         subscriber, message_class.__name__)

        if not handler:
            handler = subscriber.notify

        if subscriber not in self._subscriptions:
            self._subscriptions[subscriber] = HubCallbackContainer()

        self._subscriptions[subscriber][message_class] = handler, filter

    def is_subscribed(self, subscriber, message):
        """
        Test whether the subscriber has suscribed to a given message class

        :param subscriber: The subscriber to test
        :param message: The message class to test

        Returns:

            True if the subscriber/message pair have been subscribed to the hub

        """
        return (subscriber in self._subscriptions and
                message in self._subscriptions[subscriber])

    def get_handler(self, subscriber, message):
        if subscriber is None:
            return None
        try:
            return self._subscriptions[subscriber][message][0]
        except KeyError:
            return None

    def unsubscribe(self, subscriber, message):
        """
        Remove a (subscriber,message) pair from subscription list.
        The handler originally attached to the subscription will
        no longer be called when broadcasting messages of type message
        """
        if subscriber not in self._subscriptions:
            return
        if message in self._subscriptions[subscriber]:
            self._subscriptions[subscriber].pop(message)

    def unsubscribe_all(self, subscriber):
        """
        Unsubscribe the object from any subscriptions.
        """
        if subscriber in self._subscriptions:
            self._subscriptions.pop(subscriber)

    def _find_handlers(self, message):
        """Yields all (subscriber, handler) pairs that should receive a message
        """
        # self._subscriptions:
        # subscriber => { message type => (filter, handler)}

        # loop over subscribed objects
        for subscriber, subscriptions in list(self._subscriptions.items()):

            # subscriptions to message or its superclasses
            messages = [msg for msg in subscriptions.keys() if
                        issubclass(type(message), msg)]

            if len(messages) == 0:
                continue

            # narrow to the most-specific message
            candidate = max(messages, key=_mro_count)

            handler, test = subscriptions[candidate]
            if test(message):
                yield subscriber, handler

    @contextmanager
    def ignore_callbacks(self, ignore_type):
        self._ignore[ignore_type] += 1
        try:
            yield
        finally:
            self._ignore[ignore_type] -= 1

    @contextmanager
    def delay_callbacks(self):
        self._paused = True
        try:
            yield
        finally:
            self._paused = False
            # TODO: could de-duplicate messages here
            for message in self._queue:
                self.broadcast(message)
            self._queue = []

    def broadcast(self, message):
        """Broadcasts a message to all subscribed objects.

        :param message: The message to broadcast
        :type message: :class:`~glue.core.message.Message`
        """
        if self._ignore.get(type(message), 0) > 0:
            return
        elif self._paused:
            self._queue.append(message)
        else:
            logging.getLogger(__name__).info("Broadcasting %s", message)
            for subscriber, handler in self._find_handlers(message):
                handler(message)

    def __getstate__(self):
        """ Return a picklable representation of the hub

        Note: Only objects in glue.core are currently supported
        as pickleable. Thus, any subscriptions from objects outside
        glue.core will note be saved or restored
        """
        result = self.__dict__.copy()
        result['_subscriptions'] = self._subscriptions.copy()
        for s in self._subscriptions:
            try:
                module = s.__module__
            except AttributeError:
                module = ''
            if not module.startswith('glue.core'):
                print('Pickle warning: Hub removing subscription to %s' % s)
                result['_subscriptions'].pop(s)
        return result


class HubListener(object):

    """
    The base class for any object that subscribes to hub messages.
    This interface defines a single method, notify, that receives
    messages
    """

    def register_to_hub(self, hub):
        raise NotImplementedError

    def unregister(self, hub):
        """ Default unregistration action. Calls hub.unsubscribe_all on self"""
        hub.unsubscribe_all(self)

    def notify(self, message):
        raise NotImplementedError("Message has no handler: %s" % message)


def _mro_count(obj):
    return len(getmro(obj))
