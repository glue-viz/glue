The data framework
==================

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
components of glue about these datasets being added/removed (using the :ref:`publish_subscribe`).

For examples of how to create :class:`~glue.core.data.Data` objects and access
components, see :ref:`data_tutorial` in the user guide.

Each :class:`~glue.core.data.Data` object can optionally contain one or more subsets. These are described in more detail in :doc:`selection`.

Data factories
--------------

In order to read data into Glue, a number of built-in *data factories* are
available in :mod:`glue.data_factories`. You can easily create new data
factories by following the example in :ref:`custom_data_factory`. If the data
factory you have developed is general enough for other people to use, you can
consider adding it to the :mod:`glue.data_factories`, in which each format
should correspond to one file (e.g. ``glue/data_factories/fits.py``).

Linking?