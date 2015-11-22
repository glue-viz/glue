.. _hub:

Publish/subscribe model
==============================

.. currentmodule:: glue.core

Glue is built around a publish/subscribe paradigm that allows
individual components to remain synchronized without knowing about
each other. The central data objects in the Glue framework are:

 * :class:`~glue.core.data.Data`: Stores the actual data
 * :class:`~glue.core.subset.Subset`: Defines regions of interest in the data
 * :class:`~glue.core.data_collection.DataCollection`: Holds one or more data objects
 * :class:`~glue.core.hub.Hub`: Relays messages to other interested objects about changes in state to the data and subsets
 * :class:`~glue.core.client.Client`: Does something interesting with the data and subsets (make a plot, manipulate subsets, etc.)
 * :class:`~glue.core.message.Message`: A notice that something interesting has happened.

The typical lifecycle of these objects is as follows:

 * An empty DataCollection object is created, and automatically connected to a Hub.
 * Data are added to the data collection
 * Several clients register to the hub, and subscribe to particular
   types of messages.
 * Something (perhaps code, perhaps user interaction with a client)
   acts to change the state of a data or subset object. These changes
   automatically generate particular messages that get sent to the
   Hub. These messages communicate atomic events like Data Changed,
   Subset Changed, Subset Deleted, etc.
 * Upon receiving a message, the Hub relays it to all Clients that
   have subscribed to that particular message type.
 * The Clients react to the message however they see fit.

The documentation for these objects has more detail about
this process. However, for illustration purposes here is a simple
example of manually setting up a Glue environment:

.. literalinclude:: simple_glue.py
   :emphasize-lines: 35, 40
   :linenos:

Notice two things about this example:
 * In line 35, editing the data object automatically sends a
   DataMessage to the hub. Most message generation is handled
   automatically
 * MyClient does not recieve the message broadcast In line 40. Clients
   only receive messages they are subscribed to.
