.. _architecture:

Understanding the Core Glue Architecture
========================================

Data architecture
-----------------

The core data container in Glue is the :class:`~glue.core.data.Data` class.
Each :class:`~glue.core.data.Data` instance can include any number of
n-dimensional *components*, each represented by the
:class:`~glue.core.component.Component` class. Because of this structure, a
:class:`~glue.core.data.Data` object can represent either a table, which is a
collection of 1-d :class:`~glue.core.component.Component` objects, or an
n-dimensional dataset, which might include one (but could include more)
n-dimensional :class:`~glue.core.component.Component` objects.

When using the Glue application, the :class:`~glue.core.data.Data` objects are
collected inside a :class:`~glue.core.data_collection.DataCollection`. This
class deals not only with adding/removing datasets, but also will notify other
components of glue about these datasets being added/removed (using the `Publish/Subscribe model`_).

For examples of how to create :class:`~glue.core.data.Data` objects and access
components, see the :ref:`data_tutorial` in the user guide.

Publish/Subscribe model
-----------------------

Glue is built around a publish/subscribe paradigm that allows individual
components to remain synchronized without knowing about each other. The core
object that allows this is the :class:`~glue.core.hub.Hub`, which listens for
messages from various parts of Glue and relays messages to other interested
objects about changes in state to the data and subsets.

Messages are exchanged using :class:`~glue.core.message.Message` objects. A
message is a notice that something interesting has happened. Various
sub-classes of :class:`~glue.core.message.Message` exist, such as
:class:`~glue.core.message.DataMessage` or
:class:`~glue.core.message.SubsetMessage`.

Using the :meth:`~glue.core.hub.Hub.subscribe` method, you can easily attach callback functions/methods to specific messages using the syntax::

    hub.subscribe(self, subscriber, message_class, handler=..., filter=...)

Where the ``message_class`` is the type of message to listen for, such as
:class:`~glue.core.message.DataMessage`, the handler is the function/method to
be called if the message is received (the function/method should take one
argument which is the message), and filter can be used to specify conditions in
which to pass on the message to the function/method (for more information on
this, see the :meth:`~glue.core.hub.Hub.subscribe` documentation).

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

If we create a new dataset, then add it to the data collection created above, we then trigger the ``MyListener.receive_message`` method::

    >>> data = Data(x=[1,2,3])
    >>> data_collection.append(data)
    Message received:
    DataCollectionAddMessage:
        Sent from: DataCollection (1 data set)
        0:

Note that :class:`~glue.core.message.DataCollectionAddMessage` is a subclass of
:class:`~glue.core.message.DataCollectionMessage` - when registering to a
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

This is used in Glue to produce the following 'life-cycle':

 * An empty DataCollection object is created, and automatically connected to a
   Hub.
 * Data are added to the data collection
 * Several *clients* register to the hub, and subscribe to particular
   types of messages.
 * Something (perhaps code, perhaps user interaction with a client)
   acts to change the state of a data or subset object. These changes
   automatically generate particular messages that get sent to the
   Hub. These messages communicate atomic events like Data Changed,
   Subset Changed, Subset Deleted, etc.
 * Upon receiving a message, the Hub relays it to all clients that
   have subscribed to that particular message type.
 * The clients react to the message however they see fit.

Here, we use the term client in the generic sense of a class that interacts
with the hub. Glue includes a base :class:`~glue.core.client.Client` class that
pre-defines a number of useful connections for data viewers. Some of the data
viewers make use of this class, although there is no obligation to do so in
principle, provided the class subscribing to messages is a subclass of
:class:`~glue.core.hub.HubListener`.

Subsets and Selection
---------------------

One of the central concepts in Glue is that of subsets, which are typically
created as a result of the user selecting data in a viewer or creating the
subset from the command-line. In order to go from a selection on the screen to
defining a subset from a dataset, Glue includes the concept of a region of
interest (ROI), which is an abstract representation of a geometrical region, and subset states, which is a descriptions of the subset selection. These two concepts are described in more detail below.

Regions of interest
^^^^^^^^^^^^^^^^^^^

Basic classes for common types of ROIs are included in the :mod:`glue.core.roi`
sub-module. For example, the :mod:`~glue.core.roi.RectangularROI` class
describes a rectangular region using the lower and upper values in two
dimensions::

    >>> from glue.core.roi import RectangularROI
    >>> roi = RectangularROI(xmin=1, xmax=3, ymin=2, ymax=5)
    
Note that this is not related to any particular dataset - it is an abstract
representation of a rectangular region. All ROIs have a
:meth:`glue.core.roi.RectangularROI.contains` method that can be used to check
if a point or a set of points lies inside the region::

    >>> roi.contains(0, 3)
    False
    >>> roi.contains(2, 3)
    True
    >>> import numpy as np
    >>> x = np.array([0, 2, 4])
    >>> y = np.array([3, 3, 2])
    >>> roi.contains(x, y)
    array([False,  True, False], dtype=bool)

When a user makes a selection in a data viewer, the selection is first
translated into a ROI, after which the ROI is used to extract subsets from data. This step is done using subset states, which are described in the next section.

Subset states
-------------

While regions of interest simply define geometrical regions, subset states,
which are sub-classes of :class:`~glue.core.subset.SubsetState`, describe a
selection as a function of data component IDs. Note that this is different from
:class:`~glue.core.subset.Subset` instances, which describe the subset
*resulting* from the selection. The following simple example shows how to
easily create a :class:`~glue.core.subset.SubsetState`::

    >>> data = Data(x=[1,2,3], y=[2,3,4])
    >>> state = data.id['x'] > 1.5
    >>> state
    <InequalitySubsetState: (x > 1.5)>

Note that ``state`` is not the subset of values in ``data`` that are greater
than 1.5 -- instead, it is a representation of the inequality. This distinction
is important, because if another dataset defines a link between one of its
components and the ``x`` component of ``data``, then the inequality can be used
for that other component too.

There are different ways of applying this subset to a dataset to actually
create a subset. The easiest way of doing this is to simply call the
:meth:`~glue.core.data_collection.DataCollection.new_subset_group` method with
the :class:`~glue.core.subset.SubsetState` and a label describing that subset::

    >>> subset_group = data_collection.new_subset_group('x > 1.5', state)

This creates a :class:`~glue.core.subset_group.SubsetGroup` which represents a group of subsets, with the individual subsets accessible via the ``subsets`` attribute::

    >>> subset = subset_group.subsets[0]
    >>> subset
    Subset: x > 1.5 (data: )
    >>> subset.to_mask()
    array([False,  True,  True], dtype=bool)



While a :class:`~glue.core.component.Component`


