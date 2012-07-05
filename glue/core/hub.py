from collections import defaultdict

from .message import ErrorMessage, Message
from .exceptions import InvalidSubscriber, InvalidMessage

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
        self._subscriptions = defaultdict(dict)

        from .data import Data
        from .subset import Subset
        from .data_collection import DataCollection

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
        :type subscriber: :class:`~glue.hub.HubListener`

        :param message_class: The class of messages to subscribe to
        :type message_class: message class type (not Instance)

        :param handler:
           An optional function of the form handler(message) that will
           receive the message on behalf of the subscriber. If not provided,
           this defaults to the HubListener's notify method
        :type handler: Callable


        :param filter:
           An optional function of the form filter(message). Messages
           are only passed to the subscriber if filter(message) == True.
           The default is to always pass messages.
        :type filter: Callable


        Raises:
            InvalidMessage: If the input class isn't a
            :class:`~glue.Message` class

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
        if not handler:
            handler = subscriber.notify

        self._subscriptions[subscriber][message_class] = (filter, handler)

    def is_subscribed(self, subscriber, message):
        """
        Test whether the subscriber has suscribed to a given message class

        :param subscriber: The subscriber to test
        :param message: The message class to test

        Returns:

            True if the subscriber/message pair have been subscribed to the hub

        """
        return subscriber in self._subscriptions and \
            message in self._subscriptions[subscriber]

    def get_handler(self, subscriber, message):
        try:
            return self._subscriptions[subscriber][message][1]
        except KeyError:
            return None

    def unsubscribe(self, subscriber, message):
        """
        Remove a (subscriber,message) pair from subscription list.
        The handler originally attached to the subscription will
        no longer be called when broeacasting messages of type message
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
        for key in self._subscriptions:
            subscriber = self._subscriptions[key]
            candidates = [msg for msg in subscriber if
                          issubclass(type(message), msg)]
            if len(candidates) == 0:
                continue
            candidate = max(candidates)  # most-subclassed message class
            filter, handler = subscriber[candidate]
            if filter(message):
                yield subscriber, handler

    def broadcast(self, message):
        """Broadcasts a message to all subscribed objects.

        :param message: The message to broadcast
        :type message: :class:`~glue.message.Message`
        """
        for subscriber, handler in self._find_handlers(message):
            try:
                handler(message)
            except Exception as e:
                if isinstance(message, ErrorMessage):
                    # errors on errors! Prevent recursion
                    raise e
                else:
                    tag = str(e)
                    msg = ErrorMessage(subscriber, tag=tag)
                    self.broadcast(msg)


class HubListener(object):
    """
    The base class for any object that subscribes to hub messages.
    This interface defines a single method, notify, that receives
    messages
    """

    def register_to_hub(self, hub):
        raise NotImplementedError

    def notify(self, message):
        raise NotImplementedError("Message has no handler: %s" % message)
