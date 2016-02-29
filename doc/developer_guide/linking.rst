Linking framework
=================

One of the strengths of Glue is the ability to be able to link different
datasets together. The :ref:`linking`_ page describes how to set up links
graphically from the Glue application, but in this page, we look at how links
are set up programmatically.

As described in :ref:`data_classes`, components are identified by
:class:`~glue.core.component_id.ComponentID` instances. We can then use these
to create links across datasets. Note that links are no explicly between
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

Note that the 'from' :class:`~glue.core.component_id.ComponentID`, the first
argument, should be a list of :class:`~glue.core.component_id.ComponentID`
instances (we'll see later why that is).

Since no linking function was specified in the above example,
:class:`~glue.core.component_link.ComponentLink` defaults to the simplest kind
of link, ``identity``. For the link to be useful, we need to add it to the data
collection, and we'll be able to see what it changes::

    >>> dc.add_link(link)
    <glue.core.component.DerivedComponent at 0x10e3a6e10>

This has craated and returned a :class:`~glue.core.component.DerivedComponent`
which is a component that was derived using ``x1``, using the
:class:`~glue.core.component_link.ComponentLink` from ``x1`` to ``x2`` defined
above. If we look at the list of components on the
:class:`~glue.core.data.Data` objects, we see that the ``x2`` component in
``d2`` has been replaced by ``x1``:

    >>> print(d1.components)
    [Pixel Axis 0, World 0, x1]
    >>> print(d2.components)
    [Pixel Axis 0, World 0, x1]

This is because we used the identify transform, so since the
:class:`~glue.core.component_id.ComponentID`s ``x1`` and ``x2`` are
interchangeable, Glue decided to use ``x1`` instead of ``x2``.

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

We can instead use a custom linking function that is not simply identity::

    >>> link = ComponentLink([d1.id['x1']], d2.id['x2'],
    ...                      using=lambda x: 2*x)
    >>> dc.add_link(link)
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
    >>> diff = d3.id['xa'] - d3.id['xb']
    >>> diff
    <BinaryComponentLink: (xa - xb)>
    >>> d3['diff'] = diff
    >>> d3['diff']
    array([ 0, -1, -2])

If a component link is added to a dataset which doesn't contain the component
IDs required to compute the derived component, an ``IncompatibleAttribute``
error is raised::

    >>> d4 = Data(xc=[5,5,6])
    >>> d4.add_component_link(diff, 'diff')
    <glue.core.component.DerivedComponent at 0x10c5cb9b0>
    >>> d4['diff']
    ...
    IncompatibleAttribute: xa
    
This indicates the ``xa`` is not present in ``d4``, so the new derived
component cannot be derived in this case.