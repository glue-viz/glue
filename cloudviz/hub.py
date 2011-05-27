from collections import defaultdict

import cloudviz


class Hub(object):
    """
    The hub manages the communication between visualization clients.

    Ths hub holds references to 0 or more client objects, and 0 or 1
    translator objects. Clients subscribe to specific message
    classes. When a message is passed to the hub, the hub relays this
    message to all subscribed clients.

    Message classes are hierarchical, and all subclass from cloudviz.Message.

    Attributes
    ----------
    (XXX how to handle this under publish/subscribe paradigm?)
    translator: Translator instance, optional
    """

    def __init__(self):
        """
        Create an empty hub.
        """

        # Dictionary of client subscriptions
        self._subscriptions = defaultdict(dict)

        # Translator object will translate subsets across data sets
        self.translator = None

    def __setattr__(self, name, value):
        if name == "translator" and hasattr(self, 'translator') and \
           not isinstance(value, cloudviz.Translator):
            raise AttributeError("input is not a Translator object: %s" %
                                 type(value))
        object.__setattr__(self, name, value)

    def subscribe_client(self, client, message_class,
                         handler=None,
                         filter=lambda x: True):
        """
        Subscribe a client to a type of message class.

        The subscription is associated with an optional handler
        function, which will receive the message (the default is
        client.notify()), and a filter function, which provides extra
        control over whether the message is passed to handler.

        After subscribing, the handler will receive all messages of type
        message_class or its subclass when both of the following are true.
          - If the message is a subset of message_class, the client
            has not explicitly subscribed to any subsets of
            message_class (if so, the handler for that subscription
            receives the message)
          - The function filter(message) evaluates to true

        Parameters
        ----------
        client: HubListener instance
           The subscribing client.

        message_class: message class type (not instance)
           The class of messages to subscribe to

        handler: Function reference
           An optional function of the form handler(message) that will
           receive the message on behalf of the client. If not provided,
           this defaults to the client's notify method

        filter: Function reference
           An optional function of the form filter(message). Messages
           are only passed to the client if filter(message) == True.
           The default is to always pass messages.

        Raises
        ------
        TypeError: If the input class isn't a cloudviz.Message class,
                   or if the input client isn't a HubListener object.
        """

        if not isinstance(client, HubListener):
            raise TypeError("Client must be a HubListener: %s" %
                            type(client))

        if not isinstance(message_class, type) or \
                not issubclass(message_class, cloudviz.Message):
            raise TypeError("message class must be a subclass of "
                            "cloudviz.Message: %s" % type(message_class))
        if not handler:
            handler = client.notify

        self._subscriptions[client][message_class] = (filter, handler)

    def unsubscribe_client(self, client, message):
        if client not in self._subscriptions:
            return
        if message in self._subscriptions[client]:
            self._subscriptions[client].pop(message)

    def remove_client(self, client):
        """
        Unsubscribe the client from any subscriptions.

        Parameters
        ----------
        client: Client instance
            The client to remove

        """
        if client in self._subscriptions:
            # remove from client collection
            self._subscriptions.pop(client)

    def broadcast(self, message):
        """
        Broadcasts a message to all the clients subscribed
        to that kind of message.

        See subscribe_client for details of when a message
        is passed to the client
        """

        # loop over each (client, subscription_list) pair
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

    def translate_subset(self, subset, *args, **kwargs):
        #XXX don't know how to do this yet
        raise NotImplementedError("Translation not implemented")


class HubListener(object):
    """
    The base class for any object that subscribes to hub messages.
    This interface defines a single method, notify, that receives
    messages

    """

    def notify(self, message):
        raise NotImplementedError("Message has no handler: %s" % message)
