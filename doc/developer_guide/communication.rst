.. _communication:

The communication framework
===========================

.. _publish_subscribe:

Publish/Subscribe model
-----------------------

Glue is built around a publish/subscribe paradigm that allows individual
components to remain synchronized without knowing about each other. The core
object that allows this is the :class:`~glue.core.hub.Hub`, which listens for
messages from various parts of Glue and relays messages to other interested
objects about changes in state to the data and subsets.

You *can* instantiate a :class:`~glue.core.hub.Hub` instance directly::

    >>> from glue.core import Hub
    >>> hub = Hub()

but in most cases if you are using a
:class:`~glue.core.data_collection.DataCollection`, you can let it instantiate
the hub instead and access it via the ``.hub`` attribute::

    >>> from glue.core import DataCollection
    >>> data_collection = DataCollection()
    >>> data_collection.hub
    <glue.core.hub.Hub at 0x102991dd8>

Messages are exchanged using :class:`~glue.core.message.Message` objects. A
message is a notice that something interesting has happened. Various
sub-classes of :class:`~glue.core.message.Message` exist, such as
:class:`~glue.core.message.DataMessage` or
:class:`~glue.core.message.SubsetMessage`, and even more specialized ones such
as :class:`~glue.core.message.DataCollectionAddMessage`.

Using the :meth:`~glue.core.hub.Hub.subscribe` method, you can easily attach callback functions/methods to specific messages using the syntax::

    hub.subscribe(self, subscriber, message_class, handler=..., filter=...)

where the ``message_class`` is the type of message to listen for, such as
:class:`~glue.core.message.DataMessage`, ``handler`` is the function/method to
be called if the message is received (the function/method should take one
argument which is the message), and ``filter`` can be used to specify
conditions in which to pass on the message to the function/method (for more
information on this, see the :meth:`~glue.core.hub.Hub.subscribe`
documentation).

Subscribing to messages has to be done from a
:class:`~glue.core.hub.HubListener` instance. The following simple example shows how to set up a basic :class:`~glue.core.hub.HubListener` and register to listen for :class:`~glue.core.message.DataMessage` and :class:`~glue.core.message.DataCollectionAddMessage`::

    >>> from glue.core import Hub, HubListener, Data, DataCollection
    >>> from glue.core.message import (DataMessage,
    ...                                DataCollectionMessage)
    >>>
    >>> class MyListener(HubListener):
    ...
    ...     def __init__(self, hub):
    ...         hub.subscribe(self, DataCollectionMessage,
    ...                       handler=self.receive_message)
    ...         hub.subscribe(self, DataMessage,
    ...                       handler=self.receive_message)
    ...
    ...     def receive_message(self, message):
    ...         print("Message received:")
    ...         print("{0}".format(message))

We can then create a data collection, and create an instance of the above
class::

    >>> data_collection = DataCollection()
    >>> hub = data_collection.hub
    >>> listener = MyListener(hub)

If we create a new dataset, then add it to the data collection created above, we then trigger the ``receive_message`` method::

    >>> data = Data(x=[1,2,3])
    >>> data_collection.append(data)
    Message received:
    DataCollectionAddMessage:
        Sent from: DataCollection (1 data set)
        0:

Note that :class:`~glue.core.message.DataCollectionAddMessage` is a subclass of
:class:`~glue.core.message.DataCollectionMessage` -- when registering to a
message class, sub-classes of this message will also be received.

It is also possible to trigger messages manually::

    >>> # We can also create messages manually
    ... message = DataMessage(data)
    >>> hub.broadcast(message)
    Message received:
    DataMessage:
    	 Sent from: Data Set: Number of dimensions: 1
    Shape: 3
    Components:
     0) x
     1) Pixel Axis 0
     2) World 0

Typical workflow
----------------

This is used in Glue to produce the following communication workflow:

 * An empty :class:`~glue.core.data_collection.DataCollection` object is
   created, and automatically connected to a :class:`~glue.core.hub.Hub`.
 * Data are added to the data collection
 * Several *clients* register to the hub, and subscribe to particular
   types of messages.
 * Something (perhaps code, perhaps user interaction with a client)
   acts to change the state of a data or subset object. These changes
   automatically generate particular messages that get sent to the Hub. These
   messages communicate atomic events such as a change in the data, a change in
   a subset, or the fact a subset has been deleted.
 * Upon receiving a message, the hub relays it to all clients that
   have subscribed to that particular message type.
 * The clients react to the message however they see fit.

Here, we use the term client in the generic sense of a class that interacts
with the hub. However, Glue does include a base
:class:`~glue.core.client.Client` class that pre-defines a number of useful
connections for data viewers. Some of the data viewers make use of this class,
although there is no obligation to do so in principle, provided the class
subscribing to messages is a subclass of :class:`~glue.core.hub.HubListener`.
