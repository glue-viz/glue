.. _basedata:

Defining your own data objects
==============================

Background
----------

By default, data objects in glue are instances of the
:class:`~glue.core.data.Data` class, and this class assumes that the data are
stored in one or more local n-dimensional Numpy arrays. However, glue now
includes a way of defining a wider variety of data objects, which may rely for
example on large remote datasets, or datasets that are not inherently stored as
regular n-dimensional arrays.

The base class for all datasets is :class:`~glue.core.data.BaseData`, which is
intended to represent completely arbitrary datasets. However, glue does not yet
recognize classes that inherit directly from :class:`~glue.core.data.BaseData`.
Instead, for now, the base class that can be used to define custom data objects
is :class:`~glue.core.data.BaseCartesianData`, which inherits from
:class:`~glue.core.data.BaseData` and requires data objects to present an
interface that looks like n-dimensional arrays (although the storage of the data
could still be unstructured). In future, we will also make it possible to
support a more generic interface for data access based on the
:class:`~glue.core.data.BaseData` class.

Main data interface
-------------------

Before we dive in, we recommend that you take a look at the :ref:`data_tutorial`
tutorial to understand how the default :class:`~glue.core.data.Data` objects
work. The most important takeaway from this which is relevant here is that glue
data objects are collections of attributes (*components* in glue-speak) that are
assumed to be aligned (for regular cartesian datasets, this means they are on
the same grid). For example a table consists of a collection of 1-d attributes
that are all the same length. A traditional image consists of a single 2-d
attribute. Attributes/components are identified in data objects by
:class:`~glue.core.component_id.ComponentID` objects (though these objects don't
contain the actual values -- they are just a reference to that attribute).

To define your own data object, you should write a class that inherits from
:class:`~glue.core.data.BaseCartesianData`. You will then need to define a few
properties and methods for the data object to be usable in glue. The properties
you need to define are:

* :attr:`~glue.core.data.BaseData.label`: the name of the dataset, as a string
* :attr:`~glue.core.data.BaseCartesianData.shape`: the shape of a single
  component, given as a tuple
* :attr:`~glue.core.data.BaseData.main_components`: a list of all
  :class:`~glue.core.component_id.ComponentID` that your data object recognizes,
  excluding coordinate components (more on this later).

The methods you need to define are:

* :meth:`~glue.core.data.BaseData.get_kind`: given a
  :class:`~glue.core.component_id.ComponentID`, return a string that should be
  either ``'numerical'`` (for e.g. floating-point and integer attributes),
  ``'categorical'`` (for e.g. string attributes), or ``'datetime'`` (for
  attributes that use the ``np.datetime64`` type).
* :meth:`~glue.core.data.BaseCartesianData.get_data`: given a
  :class:`~glue.core.component_id.ComponentID` and optionally a *view*, return a
  Numpy array. A view can be anything that can be used to slice a Numpy array.
  For example a single integer (``view=4``), a tuple of slices (
  ``view=[slice(1, 14, 2), slice(4, 50, 3)]``), a list of tuples of indices
  (``view=[(1, 2, 3), (4, 3, 4)]``), and so on. If a view is specified, only that
  subset of values should be returned. For example if the data has an overall
  shape of ``(10,)`` and ``view=slice(1, 6, 2)``, ``get_data`` should
  return an array with shape ``(3,)``. By default, :meth:`BaseCartesianData.get_data <glue.core.data.BaseCartesianData.get_data>`
  will return values for pixel and world :class:`~glue.core.component_id.ComponentID`
  objects as well as any linked :class:`~glue.core.component_id.ComponentID`, so we
  recommend that your implementation calls :meth:`BaseCartesianData.get_data <glue.core.data.BaseCartesianData.get_data>`
  for any :class:`~glue.core.component_id.ComponentID` you do not expose yourself.
* :meth:`~glue.core.data.BaseCartesianData.get_mask`: given a
  :class:`~glue.core.subset.SubsetState` object (described in `Subset states`_)
  and optionally a ``view``, return a boolean array describing which values are
  in the specified subset (where `True` indicates values inside the subset).
* :meth:`~glue.core.data.BaseCartesianData.compute_statistic`: given a statistic
  name (e.g. ``'mean'``) and a :class:`~glue.core.component_id.ComponentID`, as
  well as optional keyword arguments (see
  :meth:`~glue.core.data.BaseCartesianData.compute_statistic`), return
  a statistic for the required component. In particular one of the keyword
  arguments is ``subset_state``, which can be used to indicate that the
  statistic should only be computed for a subset of values.
* :meth:`~glue.core.data.BaseCartesianData.compute_histogram`: given a list of
  :class:`~glue.core.component_id.ComponentID` objects, as well as optional
  keyword arguments (see
  :meth:`~glue.core.data.BaseCartesianData.compute_histogram`), compute an
  n-dimensional histogram of the required attributes. At the moment glue only
  makes use of this for one or two dimensions, though we may also use it for
  three dimensions in future.
* :meth:`~glue.core.data.BaseCartesianData.compute_fixed_resolution_buffer`:
  given a list of bounds of the form ``(min, max, n)`` along each dimension
  of the data, return a fixed resolution buffer that goes from the respective
  ``min`` to ``max`` in ``n`` steps. Bounds can also be single scalars in cases
  where the fixed resolution buffer is a lower-dimensional slice. This method
  can optionally take a target dataset in the case where the fixed resolution
  buffer should be computed in the frame of reference of a different dataset,
  in which case the bounds should be interpreted in the frame of reference of the
  target dataset (but this is only needed if data linking is used). See
  :meth:`~glue.core.data.BaseCartesianData.compute_fixed_resolution_buffer` for
  a full list of arguments.

Subset states
-------------

In the above section, we mentioned the concept of a
:class:`~glue.core.subset.SubsetState`. In glue parlance, a *subset state* is an
abstract representation of a subset in the data -- for instance *the subset of
data where a > 3* or *the subset of data inside a polygon with vertices vx and
vy*. Several of the methods in `Main data interface`_ can
take subset states, and so in your data object, you should decide how to
most efficiently implement each kind of subset state.

You can find a full list of subset states defined in glue :mod:`here
<glue.core.subset>`, and in particular you can look at the documentation of each
one to understand how to access the relevant information. For example, if you
have an :class:`~glue.core.subset.InequalitySubsetState`, you can access the
relevant information as follows::

    >>> subset_state
    <InequalitySubsetState: (x > 1.2)>
    >>> subset_state.left
    x
    >>> subset_state.right
    1.2
    >>> subset_state.operator
    <built-in function gt>

:class:`~glue.core.subset.SubsetState` objects have a
:meth:`~glue.core.subset.SubsetState.to_mask` method that can take a data
object and a view and return a mask::

    >>> subset_state.to_mask(d)
    array([False,  True,  True])

In this case, the subset state essentially accesses the data using
:meth:`~glue.core.data.BaseCartesianData.get_data`, so this may be very
inefficient for large datasets. Therefore, you may choose to re-interpret the
subset states and compute a mask yourself.

While developing your data class, one way to make sure that glue doesn't crash
if you haven't yet implemented support for a specific subset state is to
interpret any unimplemented subset state as simply indicating an empty subset.

Linking
-------

You should be able to link data objects that inherit from
:class:`~glue.core.data.BaseCartesianData` with other datasets - however
for this to work properly you should make sure that your implementation of
:meth:`~glue.core.data.BaseCartesianData.get_data` calls
:meth:`BaseCartesianData.get_data <glue.core.data.BaseCartesianData.get_data>`
for any unrecognized :class:`~glue.core.component_id.ComponentID`, as the base
implementation will handle returning linked values.

Using your data object
----------------------

Assuming you have written your own data class, there are several ways that you
can start using it in glue:

* You can define your own :ref:`data loader <custom_data_factory>` which is a
  function that takes a filename and should return an instance of a subclass
  of :class:`~glue.core.data.BaseCartesianData`.

* You can define your own :ref:`data importer <custom_importers>`, which is a
  function that can do anything you need, for example showing a dialog, and
  should return a list of instances of
  :class:`~glue.core.data.BaseCartesianData`. This is more general than a
  data loader since a data importer doesn't need to rely on a filename. It
  might include for example opening a dialog in which you can log in to a
  remote service and browse available datasets.

* You can start up glue programmatically, including constructing your data
  object. This is particularly useful when initially developing your custom
  data object::

    # Construct your data object
    d = MyCustomData(...)

    # Create glue application and start
    dc = DataCollection([d])
    ga = GlueApplication(dc)
    ga.start()

Example
-------

As an example of a minimal custom data class, the following implements a (very
uninteresting) dataset that simply generates values randomly in the range [0:1]
on-the-fly, and does not take subset states into account. A glue session is then
created with one of these data objects:

.. literalinclude:: random_data.py
