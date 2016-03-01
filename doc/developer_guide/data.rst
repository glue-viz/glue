The data framework
==================

.. _data_classes:

Data classes
------------

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
components of glue about these datasets being added/removed (using the :ref:`publish_subscribe` described in :ref:`communication`).

Inside :class:`~glue.core.data.Data` objects, each
:class:`~glue.core.component.Component` is assigned a
:class:`~glue.core.component_id.ComponentID`. However, this is not necesarily a
unique ID for each and every component -- instead,
different components representing the same conceptual quantity can be given the
same component ID. Component IDs are central to the linking framework
(described in :doc:`linking`). The following example shows how to access the
component IDs from a specific data object using the ``.id`` attribute::

    >>> from glue.core import Data
    >>> d = Data(x=[1,2,3], y=[2,3,4])
    >>> d.id['x']
    x
    >>> type(d.id['x'])
    Out[4]: glue.core.component_id.ComponentID

Each :class:`~glue.core.data.Data` object can optionally contain one or more subsets. These are described in more detail in :doc:`selection`.

For a more in-depth example of how to create :class:`~glue.core.data.Data`
objects and access components, see :ref:`data_tutorial` in the User guide.

Data factories
--------------

In order to read data into Glue, a number of built-in *data factories* are
available in the :mod:`glue.data_factories` module. You can easily create new
data factories by following :ref:`this <custom_data_factory>` example in the
Hacking guide . If the data factory you have developed is general enough for
other people to use, you can consider adding it to the
:mod:`glue.data_factories` module, in which each format should correspond to
one file (e.g. ``glue/data_factories/fits.py``).
