from collections import defaultdict

import cloudviz


class Hub(object):
    """
    The hub manages the communication between visualization clients,
    subsets, and other objects.

    :class:`cloudviz.hub.HubListner` objects subscribe to specific message
    classes. When a message is passed to the hub, the hub relays this
    message to all subscribers.

    Message classes are hierarchical, and all subclass from
    :class:`cloudviz.Message`.
    """

    def __init__(self, *args):
        """
        Create an empty hub.

        Inputs:
        -------
        Any dataset, subset, or client. If these are provided,
        they will automatically be registered with the new hub.
        """

        # Dictionary of subscriptions
        self._subscriptions = defaultdict(dict)

        clients = filter(lambda x: isinstance(x, cloudviz.client.Client), args)
        data = filter(lambda x: isinstance(x, cloudviz.data.Data), args)
        subsets = filter(lambda x: isinstance(x, cloudviz.subset.Subset), args)
        for d in data:
            d.register_to_hub(self)
        for c in clients:
            c.register_to_hub(self)
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
        TypeError: If the input class isn't a cloudviz.Message class,
                   or if the input subscriber isn't a HubListener object.
        """
        if not isinstance(subscriber, HubListener):
            raise TypeError("Subscriber must be a HubListener: %s" %
                            type(subscriber))
        if not isinstance(message_class, type) or \
                not issubclass(message_class, cloudviz.Message):
            raise TypeError("message class must be a subclass of "
                            "cloudviz.Message: %s" % type(message_class))
        if not handler:
            handler = subscriber.notify

        self._subscriptions[subscriber][message_class] = (filter, handler)

    def unsubscribe(self, subscriber, message):
        if subscriber not in self._subscriptions:
            return
        if message in self._subscriptions[subscriber]:
            self._subscriptions[subscriber].pop(message)

    def remove(self, subscriber):
        """
        Unsubscribe the object from any subscriptions.

        Parameters
        ----------
        subscriber: HubListener instance
            The object to remove

        """
        if subscriber in self._subscriptions:
            # remove from collection
            self._subscriptions.pop(subscriber)

    def broadcast(self, message):
        """
        Broadcasts a message to all the objects subscribed
        to that kind of message.

        See subscribe for details of when a message
        is passed to the subscriber
        """

        # loop over each (subscriber, subscription_list) pair
        for c, sub in self._subscriptions.iteritems():
            # find all messages in c's subscription list
            # that are superclasses of message
            # consider only the most-subclassed of these
            candidates = [m for m in sub if
                          issubclass(type(message), m)]
            if len(candidates) == 0:
                continue
            candidate = max(candidates)

            # only pass to handler if filter allows it
            filter, handler = sub[candidate]
            if filter(message):
                handler(message)


class HubListener(object):
    """
    The base class for any object that subscribes to hub messages.
    This interface defines a single method, notify, that receives
    messages
    """

    def notify(self, message):
        raise NotImplementedError("Message has no handler: %s" % message)
