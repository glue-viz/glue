.. _linking-framework:

The linking framework
=====================

One of the strengths of Glue is the ability to be able to link different
datasets together. The :ref:`linking` page describes how to set up links
graphically from the Glue application, but in this page, we look at how links
are set up programmatically.

Creating component links programmatically
-----------------------------------------

As described in :ref:`data_tutorial`, components are identified by
:class:`~glue.core.component_id.ComponentID` instances. We can then use these
to create links across datasets. Note that links are not defined between
:class:`~glue.core.data.Data` or :class:`~glue.core.component.Component`
objects, but between :class:`~glue.core.component_id.ComponentID` instances.

The basic linking object is :class:`~glue.core.component_link.ComponentLink`.
This describes how two :class:`~glue.core.component_id.ComponentID` instances
are linked. The following example demonstrates how to set up a
:class:`~glue.core.component_link.ComponentLink` programmatically:

   >>> from glue.core import Data, DataCollection
   >>> d1 = Data(x1=[1, 2, 3])
   >>> d2 = Data(x2=[2, 3, 4, 5])
   >>> dc = DataCollection([d1, d2])
   >>> from glue.core.component_link import ComponentLink
   >>> link = ComponentLink([d1.id['x1']], d2.id['x2'])

Note that the first
argument of :class:`~glue.core.component_link.ComponentLink` should be a list of :class:`~glue.core.component_id.ComponentID`
instances.

Since no linking function was specified in the above example,
:class:`~glue.core.component_link.ComponentLink` defaults to the simplest kind
of link, ``identity``. For the link to be useful, we need to add it to the data
collection, and we'll be able to see what it changes::

    >>> dc.add_link(link)

If we look at the list of components on the :class:`~glue.core.data.Data`
objects, we see that the ``x2`` component in ``d2`` has been replaced by ``x1``:

    >>> print(d1.components)
    [Pixel Axis 0, World 0, x1]
    >>> print(d2.components)
    [Pixel Axis 0, World 0, x1]

This is because we used the identify transform, so since the
:class:`~glue.core.component_id.ComponentID` objects ``x1`` and ``x2`` are
interchangeable, Glue decided to use ``x1`` instead of ``x2`` in ``d2`` for
simplicity.

The benefit of this is now that if we create a
:class:`~glue.core.subset.SubsetState` based on the ``x1``
:class:`~glue.core.component_id.ComponentID`, this
:class:`~glue.core.subset.SubsetState` will be applicable to both datasets:

    >>> subset_state = d2.id['x1'] > 2.5
    >>> subset_group = dc.new_subset_group('x1 > 2.5', subset_state)

This has now created subsets in both ``d1`` and ``d2``::

    >>> d1.subsets[0].to_mask()
    array([False, False,  True], dtype=bool)
    >>> d2.subsets[0].to_mask()
    array([False,  True,  True,  True], dtype=bool)

Let's now try and use a custom linking function that is not simply identity::

    >>> link = ComponentLink([d1.id['x1']], d2.id['x2'],
    ...                      using=lambda x: 2*x)
    >>> dc.add_link(link)

This time, if we look at the list of components on the :class:`~glue.core.data.Data`
objects, we see that ``d1`` now has an additional component, ``x2``::

    >>> print(d1.components)
    [Pixel Axis 0, World 0, x1, x2]
    >>> print(d2.components)
    [Pixel Axis 0, World 0, x2]

We can take a look at the values of all the components::

    >>> print(d1['x1'])
    [1 2 3]
    >>> print(d1['x2'])
    [2 4 6]
    >>> print(d2['x2'])
    [2 3 4 5]

In this case, both datasets have kept their original components, but ``d1`` now
also includes an ``x2`` :class:`~glue.core.component.DerivedComponent` which
was computed as being twice the values of ``d1['x1']``.

Creating simple component links can also be done using arithmetic operations on
:class:`~glue.core.component_id.ComponentID` instances:

    >>> d3 = Data(xa=[1, 2, 3], xb=[1, 3, 5])
    >>> dc = DataCollection([d3])
    >>> diff = d3.id['xa'] - d3.id['xb']
    >>> diff
    <BinaryComponentLink: (xa - xb)>
    >>> dc.add_link(diff)
    >>> d3['diff']
    array([ 0, -1, -2])

.. note:: This is different from using comparison operators such as ``>`` or
          ``<=`` on :class:`~glue.core.component_id.ComponentID` instances,
          which produces :class:`~glue.core.subset.SubsetState` objects.

It is also possible to add a component link to just one particular
:class:`~glue.core.data.Data` object, in which case this is equivalent to creating a :class:`~glue.core.component.DerivedComponent`. The following::

    >>> from glue.core import Data
    >>> d4 = Data(xa=[1, 2, 3], xb=[1, 3, 5])
    >>> link = d4.id['xa'] * 2
    >>> d4.add_component_link(link, 'xa_double_1')
    <glue.core.component.DerivedComponent object at 0x107b2c828>
    >>> print(d4['xa_double_1'])
    [2 4 6]

is equivalent to creating a derived component::

    >>> d4['xa_double_2'] = d4.id['xa'] * 2
    >>> print(d4['xa_double_2'])
    [2 4 6]

When adding a component link via the
:class:`~glue.core.data_collection.DataCollection`
:meth:`~glue.core.data_collection.DataCollection.add_link` method, new
component IDs are only added to :class:`~glue.core.data.Data` objects for which
the set of :class:`~glue.core.component_id.ComponentID` required for the link
already exist. For instance, in the following example, ``xu`` is only added to
``d6``::

    >>> d5 = Data(xs=[5, 5, 6])
    >>> d6 = Data(xt=[3, 2, 3])
    >>> dc = DataCollection([d5, d6])
    >>> new_component = ComponentID('xu')
    >>> link = ComponentLink([d6.id['xt']], new_component,
    ...                      using=lambda x: x + 3)
    >>> dc.add_link(link)
    >>> print(d5.components)
    [Pixel Axis 0, World 0, xs]
    >>> print(d6.components)
    [Pixel Axis 0, World 0, xt, xu]

Built-in link functions
-----------------------

Glue includes a number of built-in link functions that are collected in the
``link_function`` registry object from :mod:`glue.config`. You can easily create new link functions as described in :ref:`custom_links`, and these will then be available through the user interface, as shown in :ref:`linking` in the User guide.
