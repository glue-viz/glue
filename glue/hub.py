from collections import defaultdict

import glue
from glue.exceptions import InvalidSubscriber, InvalidMessage

class Hub(object):
    """
    The hub manages the communication between visualization clients,
    data, and other objects.

    :class:`glue.hub.HubListner` objects subscribe to specific message
    classes. When a message is passed to the hub, the hub relays this
    message to all subscribers.

    Message classes are hierarchical, and all subclass from
    :class:`glue.Message`.
    """

    def __init__(self, *args):
        """Create an empty hub.

        Inputs:
        -------

        Any dataset, subset, HubListner, or DataCollection.  If these
        are provided, they will automatically be registered with the
        new hub.
        """

        # Dictionary of subscriptions
        self._subscriptions = defaultdict(dict)

        listeners = filter(lambda x: isinstance(x, HubListener), args)
        data = filter(lambda x: isinstance(x, glue.Data), args)
        subsets = filter(lambda x: isinstance(x, glue.Subset), args)
        dcs = filter(lambda x: isinstance(x, glue.DataCollection), args)
        if len(dcs) + len(subsets) + len(data) + len(listeners) != len(args):
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
        """
        Subscribe a HubListener object to a type of message class.

        The subscription is associated with an optional handler
        function, which will receive the message (the default is
        subscriber.notify()), and a filter function, which provides extra
        control over whether the message is passed to handler.

        After subscribing, the handler will receive all messages of type
        message_class or its subclass when both of the following are true:
          - If the message is a subset of message_class, the subscriber
            has not explicitly subscribed to any subsets of
            message_class (if so, the handler for that subscription
            receives the message)
          - The function filter(message) evaluates to true

        Parameters
        ----------
        subscriber: HubListener instance
           The subscribing object.

        message_class: message class type (not instance)
           The class of messages to subscribe to

        handler: Function reference
           An optional function of the form handler(message) that will
           receive the message on behalf of the subscriber. If not provided,
           this defaults to the HubListener's notify method

        filter: Function reference
           An optional function of the form filter(message). Messages
           are only passed to the subscriber if filter(message) == True.
           The default is to always pass messages.

        Raises
        ------
        InvalidMessage: If the input class isn't a glue.Message class,
        InvalidSubscriber:
                     If the input subscriber isn't a HubListener object.
        """
        if not isinstance(subscriber, HubListener):
            raise InvalidSubscriber("Subscriber must be a HubListener: %s" %
                            type(subscriber))
        if not isinstance(message_class, type) or \
                not issubclass(message_class, glue.Message):
            raise InvalidMessage("message class must be a subclass of "
                            "glue.Message: %s" % type(message_class))
        if not handler:
            handler = subscriber.notify

        self._subscriptions[subscriber][message_class] = (filter, handler)

    def is_subscribed(self, subscriber, message):
        """
        Returns
        -------
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

        Parameters
        ----------
        subscriber:
            The object to remove
        """
        if subscriber in self._subscriptions:
            self._subscriptions.pop(subscriber)

    def _find_handlers(self, message):
        """Following the broadcast rules described above, yield all
        (subscriber, handler) pairs that should receive the message
        """
        for key in self._subscriptions:
            subscriber = self._subscriptions[key]
            candidates = [msg for msg in subscriber if
                          issubclass(type(message), msg)]
            if len(candidates) == 0:
                continue
            candidate = max(candidates) # most-subclassed message class
            filter, handler = subscriber[candidate]
            if filter(message):
                yield subscriber, handler


    def broadcast(self, message):
        """
        Broadcasts a message to all the objects subscribed
        to that kind of message.

        See subscribe for details of when a message
        is passed to the subscriber
        """
        for subscriber, handler in self._find_handlers(message):
            try:
                handler(message)
            except Exception as e:
                if isinstance(message, glue.message.ErrorMessage):
                    #prevent infinite recursion
                    break
                msg = glue.message.ErrorMessage(subscriber, tag="%s" % e)
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
